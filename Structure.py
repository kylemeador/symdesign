from __future__ import annotations

import math
import os
import subprocess
from collections import UserList, defaultdict
from collections.abc import Generator
from copy import copy
from itertools import repeat
from logging import Logger
from pathlib import Path
from random import random
from typing import IO, Sequence, Container, Literal, get_args, Callable, Any, AnyStr, Iterable

import numpy as np
from Bio.Data.IUPACData import protein_letters, protein_letters_1to3, protein_letters_3to1_extended, \
    protein_letters_1to3_extended
from scipy.spatial.transform import Rotation
from sklearn.neighbors import BallTree  # , KDTree, NearestNeighbors
from sklearn.neighbors._ball_tree import BinaryTree  # this typing implementation supports BallTree or KDTree

from PathUtils import free_sasa_exe_path, stride_exe_path, errat_exe_path, make_symmdef, free_sasa_configuration_path, \
    frag_text_file, orient_exe_path, orient_dir, reference_residues_pkl, program_name, program_version
from resources.query.pdb import get_entity_reference_sequence, retrieve_entity_id_by_sequence, query_pdb_by
from SequenceProfile import SequenceProfile, generate_mutations, get_equivalent_indices
from utils import dictionary_lookup, start_log, null_log, unpickle, digit_translate_table, remove_duplicates, \
    DesignError, ClashError, parameterize_frag_length
from classes.SymEntry import get_rot_matrices, make_rotations_degenerate
from utils.SymmetryUtils import valid_subunit_number, cubic_point_groups, point_group_symmetry_operators, \
    rotation_range, identity_matrix, origin, flip_x_matrix, valid_symmetries

# globals
logger = start_log(name=__name__)
seq_res_len = 52
coords_type_literal = Literal['all', 'backbone', 'backbone_and_cb', 'ca', 'cb', 'heavy']
directives = Literal['special', 'same', 'different', 'charged', 'polar', 'hydrophobic', 'aromatic', 'hbonding',
                     'branched']
mutation_directives: tuple[directives, ...] = get_args(directives)
atom_or_residue = Literal['atom', 'residue']
structure_container_types = Literal['atoms', 'residues', 'chains', 'entities']
termini_literal = Literal['n', 'c']
transformation_mapping: dict[str, list[float] | list[list[float]] | np.ndarray]
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
    'ARG': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'NE': 1, 'CZ': 0, 'NH1': 1, 'NH2': 1}),
    'ASN': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'OD1': 1, 'ND2': 1}),
    'ASP': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'OD1': 1, 'OD2': 1}),
    'CYS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'SG': 1}),
    'GLN': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'OE1': 1, 'NE2': 1}),
    'GLU': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'OE1': 1, 'OE2': 1}),
    'GLY': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1}),
    'HIS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'ND1': 1, 'CD2': 0, 'CE1': 0, 'NE2': 1}),
    'ILE': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG1': 0, 'CG2': 0, 'CD1': 0}),
    'LEU': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0}),
    'LYS': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0, 'CE': 0, 'NZ': 1}),
    'MET': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'SD': 1, 'CE': 0}),
    'PHE': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'CE1': 0, 'CE2': 0, 'CZ': 0,}),
    'PRO': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD': 0}),
    'SER': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'OG': 1}),
    'THR': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'OG1': 1, 'CG2': 0}),
    'TRP': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'NE1': 1, 'CE2': 0, 'CE3': 0, 'CZ2': 0, 'CZ3': 0, 'CH2': 0}),
    'TYR': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG': 0, 'CD1': 0, 'CD2': 0, 'CE1': 0, 'CE2': 0, 'CZ': 0, 'OH': 1}),
    'VAL': defaultdict(unknown_index, {'N': 1, 'CA': 0, 'C': 0, 'O': 1, 'CB': 0, 'CG1': 0, 'CG2': 0})}
hydrogens = {
    'ALA': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '3HB': 0},
    'ARG': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 0, 'HE': 1, '1HH1': 1, '2HH1': 1, '1HH2': 1, '2HH2': 1},
    'ASN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HD2': 1, '2HD2': 1,
            '1HD1': 1, '2HD1': 1},  # these are the alternative specification
    'ASP': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0},
    'CYS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HG': 1},
    'GLN': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HE2': 1, '2HE2': 1,
            '1HE1': 1, '2HE1': 1},  # these are the alternative specification
    'GLU': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0},
    'GLY': {'2HA': 0, 'H': 1, '1HA': 0, 'HA3': 0},  # last entry is from PDB version
    'HIS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 1, 'HD2': 0, 'HE1': 0, 'HE2': 1},  # this assumes HD1 is on ND1, HE2 is on NE2
    'ILE': {'H': 1, 'HA': 0, 'HB': 0, '1HG1': 0, '2HG1': 0, '1HG2': 0, '2HG2': 0, '3HG2': 0, '1HD1': 0, '2HD1': 0, '3HD1': 0,
            '3HG1': 0},  # this is the alternative specification
    'LEU': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HG': 0, '1HD1': 0, '2HD1': 0, '3HD1': 0, '1HD2': 0, '2HD2': 0, '3HD2': 0},
    'LYS': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 0, '1HE': 0, '2HE': 0, '1HZ': 1, '2HZ': 1, '3HZ': 1},
    'MET': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HE': 0, '2HE': 0, '3HE': 0},
    'PHE': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HD2': 0, 'HE1': 0, 'HE2': 0, 'HZ': 0},
    'PRO': {'HA': 0, '1HB': 0, '2HB': 0, '1HG': 0, '2HG': 0, '1HD': 0, '2HD': 1},
    'SER': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HG': 1},
    'THR': {'HA': 0, 'HB': 0, 'H': 1, 'HG1': 1, '1HG2': 0, '2HG2': 0, '3HG2': 0,
            'HG2': 1, '1HG1': 0, '2HG1': 0, '3HG1': 0},  # these are the alternative specification
    'TRP': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HE1': 1, 'HE3': 0, 'HZ2': 0, 'HZ3': 0, 'HH2': 0,  # assumes HE1 is on NE1
            'HE2': 0, 'HZ1': 0, 'HH1': 0, 'HH3': 0},  # none of these should be possible given standard nomenclature, but including incase
    'TYR': {'H': 1, 'HA': 0, '1HB': 0, '2HB': 0, 'HD1': 0, 'HD2': 0, 'HE1': 0, 'HE2': 0, 'HH': 1},
    'VAL': {'H': 1, 'HA': 0, 'HB': 0, '1HG1': 0, '2HG1': 0, '3HG1': 0, '1HG2': 0, '2HG2': 0, '3HG2': 0}}
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
            reference_sequence[chain].extend([char.title() for char in sequence])
        else:
            reference_sequence[chain] = [char.title() for char in sequence]

    # Ensure we parse selenomethionine correctly
    protein_letters_3to1_extended_mse = protein_letters_3to1_extended.copy()
    protein_letters_3to1_extended_mse['Mse'] = 'M'

    # Format the sequences as a one AA letter list
    reference_sequences = {}  # []
    for chain, sequence in reference_sequence.items():
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
    reference_sequence = parse_seqres(seq_res_lines)
    for entity_name, info in entity_info.items():
        # Grab the first chain from the identified chains, and use it to grab the reference sequence
        chain = info['chains'][0]
        info['reference_sequence'] = reference_sequence[chain]  # Used when parse_seqres returns dict[str, str]
        info['dbref'] = dbref[chain]

    # Convert the incrementing reference sequence to a list of the sequences
    reference_sequence = list(reference_sequence.values())

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


class Coords:
    """Responsible for handling StructureBase coordinates by storing in a numpy.ndarray with shape (n, 3) where n is the
     number of atoms in the structure and the 3 dimensions represent x, y, and z coordinates

    Args:
        coords: The coordinates to store. If none are passed an empty container will be generated
    """
    coords: np.ndarray

    def __init__(self, coords: np.ndarray | list[list[float]] = None):
        if coords is None:
            self.coords = np.array([])
        elif not isinstance(coords, (np.ndarray, list)):
            raise TypeError(f'Can\'t initialize {type(self).__name__} with {type(coords).__name__}. Type must be a '
                            f'numpy.ndarray of float with shape (n, 3) or list[list[float]]')
        else:
            self.coords = np.array(coords, np.float_)

    def delete(self, indices: Sequence[int]):
        """Delete coordinates from the Coords container

        Args:
            indices: The indices to delete from the Coords array
        Sets:
            self.coords = numpy.delete(self.coords, indices)
        """
        self.coords = np.delete(self.coords, indices, axis=0)

    def insert(self, new_coords: np.ndarray | list[list[float]], at: int = None):
        """Insert additional coordinates into the Coords container

        Args:
            new_coords: The coords to include into Coords
            at: The index to perform the insert at
        Sets:
            self.coords = numpy.concatenate(self.coords[:at] + new_coords + self.coords[at:])
        """
        self.coords = \
            np.concatenate((self.coords[:at] if 0 <= at <= len(self.coords) else self.coords, new_coords,
                            self.coords[at:])
                           if at else (self.coords[:at] if 0 <= at <= len(self.coords) else self.coords, new_coords))

    def replace(self, indices: Sequence[int], new_coords: np.ndarray | list[list[float]]):
        """Replace existing coordinates in the Coords container with new coordinates

        Args:
            indices: The indices to delete from the Coords array
            new_coords: The coordinate values to replace in Coords
        Sets:
            self.coords[indices] = new_coords
        """
        try:
            self.coords[indices] = new_coords
        except ValueError as error:  # they are probably different lengths or another numpy indexing/setting issue
            if self.coords.shape[0] == 0:  # there are no coords, lets use set mechanism
                self.coords = new_coords
            else:
                raise ValueError(f'The new_coords are not the same shape as the selected indices {error}')

    def set(self, coords: np.ndarray | list[list[float]]):
        """Set self.coords to the provided coordinates

        Args:
            coords: The coordinate values to set
        Sets:
            self.coords = coords
        """
        self.coords = coords

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __iter__(self) -> list[float, float, float]:
        yield from self.coords.tolist()

    def __copy__(self):  # -> Self Todo python3.11
        other = self.__class__.__new__(self.__class__)
        # other.__dict__ = self.__dict__.copy()
        other.coords = self.coords.copy()

        return other


# null_coords = Coords()
# parent Structure controls these attributes
parent_variable = '_StructureBase__parent'
new_parent_attributes = ('_coords', '_log', '_atoms', '_residues')
parent_attributes = (parent_variable,) + new_parent_attributes
"""Holds all the attributes which the parent StructureBase controls"""


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
        """Query whether the Structure is symmetric"""
        try:
            return self.symmetry is not None
        except AttributeError:
            return False


class StructureBase(Symmetry):
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
                 , header=None, biological_assembly=None, cryst_record=None, entity_info=None, multimodel=None,
                 resolution=None, reference_sequence=None, sequence=None, entities=None,
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
                    raise TypeError(f'Can\'t set Log to {type(log).__name__}. Must be type logging.Logger')
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
            raise TypeError(f'The argument(s) passed to the StructureBase object were not recognized and aren\'t '
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
            raise TypeError(f'Can\'t set Log to {type(log).__name__}. Must be type logging.Logger')

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
        if self.is_parent() and self.is_symmetric() and self._symmetric_dependents:
            # This Structure is a symmetric parent, update dependent coords to update the parent
            self.log.debug(f'self._symmetric_dependents: {self._symmetric_dependents}')
            for dependent in self._symmetric_dependents:
                if dependent.is_symmetric():
                    dependent._parent_is_updating = True
                    self.log.debug(f'Setting {dependent.name} _symmetric_dependent coords')
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

    def __init__(self, index: int = None, number: int = None, atom_type: str = None, alt_location: str = None,
                 residue_type: str = None, chain: str = None, residue_number: int = None,
                 code_for_insertion: str = None, coords: list[float] = None, occupancy: float = None,
                 b_factor: float = None, element: str = None, charge: str = None, **kwargs):
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
        self.__coords = coords if coords else []
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
        # number, atom_type, alt_location, residue_type, chain, residue_number, code_for_insertion, occupancy, b_factor,
        # element, charge
        """Initialize without coordinates"""
        return cls(index=idx, number=number, atom_type=atom_type, alt_location=alt_location, residue_type=residue_type,
                   chain=chain, residue_number=residue_number, code_for_insertion=code_for_insertion,
                   occupancy=occupancy, b_factor=b_factor, element=element, charge=charge)

    def detach_from_parent(self):
        """Remove the current instance from the parent that created it"""
        setattr(self, parent_variable, None)  # set parent explicitly as None
        # # Extract the coordinates
        # coords = self.coords
        # create a new, empty Coords instance
        self._coords = Coords(self.coords)
        self.index = 0

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

    def return_atom_record(self) -> str:
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

    def insert(self, new_atoms: list[Atom] | np.ndarray, at: int = None):
        """Insert Atom objects into the Atoms container

        Args:
            new_atoms: The residues to include into Residues
            at: The index to perform the insert at
        """
        self.atoms = np.concatenate((self.atoms[:at] if 0 <= at <= len(self.atoms) else self.atoms,
                                     new_atoms if isinstance(new_atoms, Iterable) else [new_atoms],
                                     self.atoms[at:] if at is not None else []))

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def atoms(self) -> list[Atom] | None:
        """Return the Atom instances in the Structure"""
        try:
            return self._atoms.atoms[self._atom_indices].tolist()
        except AttributeError:  # when self._atoms isn't set or is None and doesn't have .atoms
            return

    @atoms.setter
    def atoms(self, atoms: Atoms | list[Atom]):
        """Set the Structure atoms to an Atoms object"""
        # Todo make this setter function in the same way as self._coords.replace?
        if isinstance(atoms, Atoms):
            self._atoms = atoms
        else:
            self._atoms = Atoms(atoms)

    # # Todo enable this type of functionality
    # @atoms.setter
    # def atoms(self, atoms: Atoms):
    #     self._atoms.replace(self._atom_indices, atoms)

    # Todo create add_atoms that is like list append
    def add_atoms(self, atom_list):
        """Add Atoms in atom_list to the Structure instance"""
        raise NotImplementedError('This function (add_atoms) is currently broken')
        atoms = self.atoms.tolist()
        atoms.extend(atom_list)
        self.atoms = atoms
        # Todo need to update all referrers
        # Todo need to add the atoms to coords

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

    def _assign_atoms(self, atoms: Atoms | list[Atom], atoms_only: bool = True,
                      **kwargs):  # same function in Residue
        """Assign Atom instances to the Structure, create Atoms object, and create Residue instances/Residues

        Args:
            atoms: The Atom instances to assign to the Structure
            atoms_only: Whether Atom instances are being assigned on their own. Residues will be created if so.
                If not, indicate False and use other Structure information such as Residue instances to complete set up.
                When False, atoms won't become dependents of this instance until specifically called using
                Atoms.set_attributes(_parent=self)
        Keyword Args:
            coords=None (numpy.ndarray): The coordinates to assign to the Structure.
                Optional, will use Residues.coords if not specified
        Sets:
            self._atom_indices (list[int])

            self._atoms (Atoms)
        """
        # set proper atoms attributes
        self._atom_indices = list(range(len(atoms)))
        if not isinstance(atoms, Atoms):  # must create the Atoms object
            atoms = Atoms(atoms)

        if atoms.are_dependents():  # copy Atoms object to set new attributes on each member Atom
            atoms = copy(atoms)
            atoms.reset_state()  # clear runtime attributes
        self._atoms = atoms
        self.renumber_atoms()

        if atoms_only:
            self._populate_coords(**kwargs)  # coords may be passed
            # self._create_residues()
            # ensure that coordinate lengths match atoms
            self._validate_coords()
            # update Atom instance attributes to ensure they are dependants of this instance
            # must do this after _populate_coords to ensure that coordinate info isn't overwritten
            self._atoms.set_attributes(_parent=self)
            # if not self.file_path:  # assume this instance wasn't parsed and Atom indices are incorrect
            self._atoms.reindex()
            # self._set_coords_indexed()

    def _populate_coords(self, from_source: structure_container_types = 'atoms', coords: np.ndarray = None):
        """Set up the coordinates, initializing them from_source coords if none are set

        Only useful if the calling Structure is a parent, and coordinate initialization has yet to occur
        Args:
            from_source: The source to set the coordinates from if they are missing
            coords: The coordinates to assign to the Structure. Optional, will use from_source.coords if not specified
        """
        if coords:  # try to set the provided coords. This will handle issue where empty Coords class should be set
            # Setting .coords through normal mechanism preserves subclasses requirement to handle symmetric coordinates
            self.coords = np.concatenate(coords)
        if self._coords.coords.shape[0] == 0:  # check if Coords (_coords) hasn't been populated
            # if it hasn't, then coords weren't passed. try to set from self.from_source. catch missing from_source
            try:
                self._coords.set(np.concatenate([s.coords for s in getattr(self, from_source)]))
            except AttributeError:
                try:  # probably missing from_source. .coords is available in all structure_container_types...
                    getattr(self, from_source)
                except AttributeError:
                    raise AttributeError(f'{from_source} is not set on the current {type(self).__name__} instance!')
                raise AttributeError(f'Missing .coords attribute on the current {type(self).__name__} '
                                     f'instance.{from_source} attribute. This is really not supposed to happen! '
                                     f'Congrats you broke a core feature! 0.15 bitcoin have been added to your wallet')

    def _validate_coords(self):
        """Ensure that the StructureBase coordinates are formatted correctly"""
        # this is the functionality we are testing most of the time
        if self.number_of_atoms != len(self.coords):  # number_of_atoms was just set by self._atom_indices
            raise ValueError(f'The number of Atoms ({self.number_of_atoms}) != number of Coords ({len(self.coords)}). '
                             f'Consider initializing {type(self).__name__} without explicitly passing coords if this '
                             f'isn\'t expected')

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
            f'HEADER                                            21-AUG-21   XXXX              \n' \
            f'EXPDTA    THEORETICAL MODEL                                                     \n' \
            f'REMARK 220                                                                      \n' \
            f'REMARK 220 EXPERIMENTAL DETAILS                                                 \n' \
            f'REMARK 220  EXPERIMENT TYPE                : THEORETICAL MODELLING              \n' \
            f'REMARK 220  DATE OF DATA COLLECTION        : 21-AUG-21                          \n' \
            f'REMARK 220                                                                      \n' \
            f'REMARK 220 REMARK: MODEL GENERATED BY {program_name:<42s}\n' \
            f'REMARK 220  VERSION {program_version:<60s}\n'

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
            file_handle.write(f'{self.return_atom_record(**kwargs)}\n')
            return None
        else:  # out_path always has default argument current working directory
            _header = self.format_header(**kwargs)
            if header is not None and isinstance(header, str):
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                outfile.write(f'{self.return_atom_record(**kwargs)}\n')
            return out_path


class GhostFragment:
    _guide_coords: np.ndarray
    """The guide coordinates according to the representative ghost fragment"""
    _representative: Structure
    aligned_fragment: Fragment
    """Must support .chain, .number, and .transformation attributes"""
    fragment_db: object  # Todo typing with FragmentDatabase
    i_type: int
    j_type: int
    k_type: int
    rmsd: float
    """The deviation from the representative ghost fragment"""

    def __init__(self, guide_coords: np.ndarray, i_type: int, j_type: int, k_type: int, ijk_rmsd: float,
                 aligned_fragment: Fragment):
        self._guide_coords = guide_coords
        self.i_type = i_type
        self.j_type = self.frag_type = j_type
        self.k_type = k_type
        self.rmsd = ijk_rmsd
        self.aligned_fragment = aligned_fragment
        self.fragment_db = aligned_fragment.fragment_db

    @property
    def type(self) -> int:
        """The secondary structure of the Fragment"""
        return self.j_type

    # @type.setter
    # def type(self, frag_type: int):
    #     """Set the secondary structure of the Fragment"""
    #     self.j_type = frag_type

    # @property
    # def frag_type(self) -> int:
    #     """The secondary structure of the Fragment"""
    #     return self.j_type

    # @frag_type.setter
    # def frag_type(self, frag_type: int):
    #     """Set the secondary structure of the Fragment"""
    #     self.j_type = frag_type

    @property
    def ijk(self) -> tuple[int, int, int]:
        """The Fragment cluster index information

        Returns:
            I cluster index, J cluster index, K cluster index
        """
        return self.i_type, self.j_type, self.k_type

    @property
    def get_aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the GhostFragment was mapped to

        Returns:
            aligned chain, aligned residue_number
        """
        return self.aligned_fragment.chain, self.aligned_fragment.number

    @property
    def number(self) -> int:
        """The Residue number of the aligned Fragment"""
        return self.aligned_fragment.number

    @property
    def guide_coords(self) -> np.ndarray:
        """Return the guide coordinates of the GhostFragment"""
        rotation, translation = self.aligned_fragment.transformation  # self.transformation
        return np.matmul(self._guide_coords, np.transpose(rotation)) + translation

    @property
    def rotation(self) -> np.ndarray:
        """The rotation of the aligned Fragment from the Fragment Database"""
        return self.aligned_fragment.rotation

    @property
    def translation(self) -> np.ndarray:
        """The rotation of the aligned Fragment from the Fragment Database"""
        return self.aligned_fragment.translation

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the aligned Fragment from the Fragment Database

        Returns:
            The rotation (3, 3), the translation (3,)
        """
        return self.aligned_fragment.transformation

    @property
    def representative(self) -> Structure:
        """Access the Representative GhostFragment Structure"""
        try:
            return self._representative.return_transformed_copy(*self.transformation)
        except AttributeError:
            self._representative, _ = dictionary_lookup(self.fragment_db.paired_frags, self.ijk)

        return self._representative.return_transformed_copy(*self.transformation)

    def write(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None, **kwargs) -> \
            str | None:
        """Write the GhostFragment to a file specified by out_path or with a passed file_handle

        If a file_handle is passed, no header information will be written. Arguments are mutually exclusive
        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
        """
        if file_handle:
            file_handle.write(f'{self.representative.return_atom_record(**kwargs)}\n')
            return None
        else:  # out_path always has default argument current working directory
            _header = self.representative.format_header(**kwargs)
            if header is not None and isinstance(header, str):  # used for cryst_record now...
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)
                outfile.write(f'{self.representative.return_atom_record(**kwargs)}\n')
            return out_path

    # def get_center_of_mass(self):  # UNUSED
    #     return np.matmul(np.array([0.33333, 0.33333, 0.33333]), self.guide_coords)


class Fragment:
    _fragment_ca_coords: np.ndarray
    _representative_ca_coords: np.ndarray
    chain: str
    frag_lower_range: int
    frag_upper_range: int
    fragment_db: object  # Todo typing with FragmentDatabase
    ghost_fragments: list | list[GhostFragment] | None
    # guide_coords: np.ndarray | None
    i_type: int | None
    number: int
    rmsd_thresh: float = 0.75
    rotation: np.ndarray
    template_coords = np.array([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.]])
    translation: np.ndarray

    def __init__(self, fragment_type: int = None,
                 # guide_coords: np.ndarray = None,
                 # fragment_length: int = 5,
                 fragment_db: object = None,
                 # fragment_db: FragmentDatabase = None,  # Todo typing with FragmentDatabase
                 **kwargs):
        self.ghost_fragments = None
        self.i_type = fragment_type
        # self.guide_coords = guide_coords
        # self.fragment_length = fragment_length
        self.rotation = identity_matrix
        self.translation = origin
        # if fragment_db is not None:
        self.fragment_db = fragment_db
        # self.frag_lower_range, self.frag_upper_range = fragment_db.fragment_range
        super().__init__(**kwargs)
        # may need FragmentBase to clean extras for proper method resolution order (MRO)

    @property
    def fragment_db(self) -> object | None:
        """The secondary structure of the Fragment"""
        return self._fragment_db

    @fragment_db.setter
    def fragment_db(self, fragment_db: object):  # Todo typing with FragmentDatabase
        """Set the secondary structure of the Fragment"""
        self._fragment_db = fragment_db
        if fragment_db is not None:
            self.frag_lower_range, self.frag_upper_range = fragment_db.fragment_range
            self.fragment_length = fragment_db.fragment_length

    @property
    def frag_type(self) -> int | None:
        """The secondary structure of the Fragment"""
        return self.i_type

    @frag_type.setter
    def frag_type(self, frag_type: int):
        """Set the secondary structure of the Fragment"""
        self.i_type = frag_type

    @property
    def get_aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the MonoFragment was mapped to

        Returns:
            aligned chain, aligned residue_number
        """
        return self.chain, self.number

    @property
    def _representative_coords(self) -> np.ndarray:
        """Return the CA coordinates of the mapped fragment"""
        try:
            return self._representative_ca_coords
        except AttributeError:
            self._representative_ca_coords = self.fragment_db.reps[self.i_type]
            return self._representative_ca_coords

    @_representative_coords.setter
    def _representative_coords(self, coords: np.ndarray):
        self._representative_ca_coords = coords

    @property
    def guide_coords(self) -> np.ndarray:
        """Return the guide coordinates of the mapped Fragment"""
        rotation, translation = self.transformation  # This updates the transformation on the fly if possible
        return np.matmul(self.template_coords, np.transpose(rotation)) + translation
        # return np.matmul(self.template_coords, np.transpose(self.rotation)) + self.translation

    # @guide_coords.setter
    # def guide_coords(self, coords: np.ndarray):
    #     self.guide_coords = coords

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the Fragment from the FragmentDatabase to its current position"""
        # return dict(rotation=self.rotation, translation=self.translation)
        return self.rotation, self.translation
        # return dict(rotation=self.rotation, translation=self.translation)

    # def get_center_of_mass(self):  # UNUSED
    #     if self.guide_coords:
    #         return np.matmul([0.33333, 0.33333, 0.33333], self.guide_coords)
    #     else:
    #         return None

    def find_ghost_fragments(self,
                             # indexed_ghost_fragments: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
                             clash_tree: BinaryTree = None, clash_dist: float = 2.2):
        """Find all the GhostFragments associated with the Fragment

        Args:
            clash_tree: Allows clash prevention during search. Typical use is the backbone and CB atoms of the
                Structure that the Fragment is assigned
            clash_dist: The distance to check for backbone clashes
        Returns:
            The ghost fragments associated with the fragment
        """
        #             indexed_ghost_fragments: The paired fragment database to match to the Fragment instance
        # ghost_i_type = indexed_ghost_fragments.get(self.i_type, None)

        # self.fragment_db.indexed_ghosts : dict[int,
        #                                        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        ghost_i_type_arrays = self.fragment_db.indexed_ghosts.get(self.i_type, None)
        if ghost_i_type_arrays is None:
            self.ghost_fragments = []
            return

        stacked_bb_coords, stacked_guide_coords, ijk_types, rmsd_array = ghost_i_type_arrays
        # transformed_guide_coords = transform_coordinate_sets(stacked_guide_coords, *self.transformation)
        if clash_tree is None:
            viable_indices = None
        else:
            transformed_bb_coords = transform_coordinate_sets(stacked_bb_coords, *self.transformation)
            # with .reshape(), we query on a np.view saving memory
            neighbors = clash_tree.query_radius(transformed_bb_coords.reshape(-1, 3), clash_dist)
            neighbor_counts = np.array([neighbor.size for neighbor in neighbors])
            # reshape to original size then query for existence of any neighbors for each fragment individually
            clashing_indices = neighbor_counts.reshape(transformed_bb_coords.shape[0], -1).any(axis=1)
            viable_indices = ~clashing_indices

        # self.ghost_fragments = [GhostFragment(*info) for info in zip(list(transformed_guide_coords[viable_indices]),
        self.ghost_fragments = [GhostFragment(*info) for info in zip(list(stacked_guide_coords[viable_indices]),
                                                                     *zip(*ijk_types[viable_indices].tolist()),
                                                                     rmsd_array[viable_indices].tolist(), repeat(self))]

    def get_ghost_fragments(self,
                            # indexed_ghost_fragments: dict,
                            **kwargs) -> list | list[GhostFragment]:
        """Find and return all the GhostFragments associated with the Fragment. Optionally check clashing with the
        original structure backbone

        Keyword Args:
            clash_tree: sklearn.neighbors._ball_tree.BinaryTree = None - Allows clash prevention during search.
                Typical use is the backbone and CB coordinates of the Structure that the Fragment is assigned
            clash_dist: float = 2.2 - The distance to check for backbone clashes
        Returns:
            The ghost fragments associated with the fragment
        """
        #         Args:
        #             indexed_ghost_fragments: The paired fragment database to match to the Fragment instance
        self.find_ghost_fragments(**kwargs)
        return self.ghost_fragments

    # def __copy__(self):  # -> Self # Todo python3.11
    #     other = self.__class__.__new__(self.__class__)
    #     other.__dict__ = copy(self.__dict__)
    #     other.__dict__['ghost_fragments'] = copy(self.ghost_fragments)


class MonoFragment(Fragment):
    """Used to represent Fragment information when treated as a continuous Structure Fragment of length fragment_length
    """
    _fragment_coords: np.ndarray  # This is a property in ResidueFragment
    central_residue: Residue

    def __init__(self, residues: Sequence[Residue],
                 fragment_db: object = None,
                 # fragment_db: FragmentDatabase = None,  # Todo typing with FragmentDatabase
                 **kwargs):
        super().__init__(**kwargs)
        self.central_residue = residues[int(self.fragment_length/2)]

        if not residues:
            raise ValueError(f'Can\'t find {type(self).__name__} without passing residues with length '
                             f'{self.fragment_length}')
        elif fragment_db is None:
            raise ValueError(f"Can't find {type(self).__name__} without passing fragment_db")
        else:
            try:
                representatives: dict[int, np.ndarray] = fragment_db.reps
            except AttributeError:
                raise TypeError(f'The passed fragment_db is not of the required type "FragmentDatabase"')

        self._fragment_coords = np.array([residue.ca_coords for residue in residues])
        min_rmsd = float('inf')
        for cluster_type, cluster_coords in representatives.items():
            rmsd, rot, tx = superposition3d(self._fragment_coords, cluster_coords)
            if rmsd <= self.rmsd_thresh and rmsd <= min_rmsd:
                self.i_type = cluster_type
                min_rmsd, self.rotation, self.translation = rmsd, rot, tx

        if self.i_type:
            # self.guide_coords = \
            #     np.matmul(self.template_coords, np.transpose(self.rotation)) + self.translation
            self._representative_ca_coords = representatives[self.i_type]

    @property
    def get_aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the MonoFragment was mapped to

        Returns:
            aligned chain, aligned residue_number
        """
        return self.central_residue.chain, self.central_residue.number

    @property
    def chain(self) -> str:
        """The Residue number"""
        return self.central_residue.chain

    @property
    def number(self) -> int:
        """The Residue number"""
        return self.central_residue.number

    # Methods below make compatible with Pose symmetry operations
    @property
    def coords(self) -> np.ndarray:
        return self.guide_coords

    @coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        if coords.shape == (3, 3):
            # Move the transformation accordingly
            _, self.rotation, self.translation = superposition3d(coords, self.template_coords)
            # self.guide_coords = coords
        else:
            raise ValueError(f'{type(self).__name__} coords must be shape (3, 3), not {coords.shape}')

    # def return_transformed_copy(self, rotation: list | np.ndarray = None, translation: list | np.ndarray = None,
    #                             rotation2: list | np.ndarray = None, translation2: list | np.ndarray = None) -> \
    #         MonoFragment:
    #     """Make a semi-deep copy of the Structure object with the coordinates transformed in cartesian space
    #
    #     Transformation proceeds by matrix multiplication with the order of operations as:
    #     rotation, translation, rotation2, translation2
    #
    #     Args:
    #         rotation: The first rotation to apply, expected array shape (3, 3)
    #         translation: The first translation to apply, expected array shape (3,)
    #         rotation2: The second rotation to apply, expected array shape (3, 3)
    #         translation2: The second translation to apply, expected array shape (3,)
    #     Returns:
    #         A transformed copy of the original object
    #     """
    #     if rotation is not None:  # required for np.ndarray or None checks
    #         new_coords = np.matmul(self.guide_coords, np.transpose(rotation))
    #     else:
    #         new_coords = self.guide_coords
    #
    #     if translation is not None:  # required for np.ndarray or None checks
    #         new_coords += np.array(translation)
    #
    #     if rotation2 is not None:  # required for np.ndarray or None checks
    #         new_coords = np.matmul(new_coords, np.transpose(rotation2))
    #
    #     if translation2 is not None:  # required for np.ndarray or None checks
    #         new_coords += np.array(translation2)
    #
    #     new_structure = copy(self)
    #     new_structure.guide_coords = new_coords
    #
    #     return new_structure


class ResidueFragment(Fragment):
    """Represent Fragment information for a single Residue"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # @property
    # def frag_type(self) -> int | None:
    #     """The secondary structure of the Fragment"""
    #     return self.i_type
    #
    # @frag_type.setter
    # def frag_type(self, frag_type: int):
    #     """Set the secondary structure of the Fragment"""
    #     self.i_type = frag_type

    @property
    def _fragment_coords(self) -> np.ndarray:
        """Return the CA coordinates of the neighboring Residues which specify the ResidueFragment"""
        # try:
        #     return self._fragment_ca_coords
        # except AttributeError:
        return np.array([res.ca_coords for res in self.get_upstream(self.frag_lower_range)
                         + [self] + self.get_downstream(self.frag_upper_range-1)])
        #     return self._fragment_ca_coords

    # @_fragment_coords.setter
    # def _fragment_coords(self, coords: np.ndarray):
    #     self._fragment_ca_coords = coords

    @property
    def transformation(self) -> tuple[np.ndarray, np.ndarray]:  # dict[str, np.ndarray]:
        """The transformation of the Fragment from the FragmentDatabase to its current position"""
        # return dict(rotation=self.rotation, translation=self.translation)
        _, self.rotation, self.translation = superposition3d(self._fragment_coords, self._representative_coords)
        return self.rotation, self.translation
        # return dict(rotation=self.rotation, translation=self.translation)

    @property
    def get_aligned_chain_and_residue(self) -> tuple[str, int]:
        """Return the Fragment identifiers that the ResidueFragment was mapped to

        Returns:
            aligned chain, aligned residue_number
        """
        return self.chain, self.number


class Residue(ResidueFragment, ContainsAtomsMixin):
    _ca_indices: list[int]
    _cb_indices: list[int]
    _bb_indices: list[int]
    _bb_and_cb_indices: list[int]
    _heavy_atom_indices: list[int]
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
        {'_secondary_structure', '_sasa', '_sasa_aploar', '_sasa_polar', '_contact_order', '_local_density'}
    type: str

    def __init__(self, atoms: list[Atoms] | Atoms = None, atom_indices: list[int] = None, **kwargs):
        # kwargs passed to StructureBase
        #          parent: StructureBase = None, log: Log | Logger | bool = True, coords: list[list[float]] = None
        # kwargs passed to ResidueFragment -> Fragment
        #          fragment_type: int = None, guide_coords: np.ndarray = None, fragment_length: int = 5,
        super().__init__(**kwargs)
        # Unused args now
        #  atoms: Atoms = None,
        #        index=None
        # self.index = index

        parent = self.parent
        if parent:  # we are setting up a dependent Residue
            self._atom_indices = atom_indices
            try:
                self._start_index = atom_indices[0]
            except (TypeError, IndexError):
                raise IndexError('The Residue wasn\'t passed atom_indices which are required for initialization')
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
            return None

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
            raise ValueError('Can\'t get 0 upstream residues. 1 or more must be specified')

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
            raise ValueError('Can\'t get 0 downstream residues. 1 or more must be specified')

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
            for atom in self.atoms:
                polarity_list[residue_atom_polarity.get(atom.type)].append(atom.sasa)
        except AttributeError:  # missing atom.sasa
            self.parent.get_sasa()
            for atom in self.atoms:
                polarity_list[residue_atom_polarity.get(atom.type)].append(atom.sasa)

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

    def return_atom_record(self, **kwargs) -> str:
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

        if self.is_parent():  # this Structure is the parent, it's copy should be too
            # set the copying Structure attribute ".spawn" to indicate to dependents the "other" of this copy
            self.spawn = other
            try:
                for attr in parent_attributes:
                    other.__dict__[attr] = copy(self.__dict__[attr])
            except KeyError:  # '_residues' is not present and will be last
                pass
            other._atoms.set_attributes(_parent=other)  # Todo comment out when Atom.__copy__ needed somewhere
            # remove the attribute spawn after other Structure containers are copied
            del self.spawn
        else:  # this Structure is a dependent, it's copy should be too
            try:
                other._parent = self.parent.spawn
            except AttributeError:  # this copy was initiated by a Structure that is not the parent
                # self.log.debug(f'The copied {type(self).__name__} is being set as a parent. It was a dependent '
                #                f'previously')
                if self._copier:  # Copy initiated by Residues container
                    pass
                else:
                    other.detach_from_parent()

        return other


class Residues:
    residues: np.ndarray

    def __init__(self, residues: list[Residue] | np.ndarray = None):
        if residues is None:
            self.residues = np.array([])
        elif not isinstance(residues, (np.ndarray, list)):
            raise TypeError(f'Can\'t initialize {type(self).__name__} with {type(residues).__name__}. Type must be a '
                            f'numpy.ndarray or list of {Residue.__name__} instances')
        else:
            self.residues = np.array(residues, dtype=np.object_)
        self.find_prev_and_next()

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

    def reindex_atoms(self, start_at: int = 0):  # , offset=None):
        """Set each member Residue indices according to incremental Atoms/Coords index

        Args:
            start_at: The integer to start renumbering Residue, Atom objects at
        """
        residue: Residue
        if start_at > 0:
            if start_at < self.residues.shape[0]:  # if in the Residues index range
                prior_residue = self.residues[start_at - 1]
                # prior_residue.start_index = start_at
                for residue in self.residues[start_at:].tolist():
                    residue.start_index = prior_residue.atom_indices[-1] + 1
                    prior_residue = residue
            else:
                # self.residues[-1].start_index = self.residues[-2].atom_indices[-1] + 1
                raise IndexError(f'{Residues.reindex_atoms.__name__}: Starting index is outside of the '
                                 f'allowable indices in the Residues object!')
        else:  # when start_at is 0 or less
            prior_residue = self.residues[0]
            prior_residue.start_index = start_at
            for residue in self.residues[1:].tolist():
                residue.start_index = prior_residue.atom_indices[-1] + 1
                prior_residue = residue

    def insert(self, new_residues: list[Residue] | np.ndarray, at: int = None):
        """Insert Residue(s) into the Residues object

        Args:
            new_residues: The residues to include into Residues
            at: The index to perform the insert at
        """
        self.residues = np.concatenate((self.residues[:at] if 0 <= at <= len(self.residues) else self.residues,
                                        new_residues if isinstance(new_residues, Iterable) else [new_residues],
                                        self.residues[at:] if at is not None else []))

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

        other.find_prev_and_next()

        return other

    def __len__(self) -> int:
        return self.residues.shape[0]

    def __iter__(self) -> Residue:
        yield from self.residues.tolist()


def write_frag_match_info_file(ghost_frag: GhostFragment = None, matched_frag: Fragment = None,
                               overlap_error: float = None, match_number: int = None,
                               central_frequencies=None, out_path: AnyStr = os.getcwd(), pose_id: str = None):
    # ghost_residue: Residue = None, matched_residue: Residue = None,

    # if not ghost_frag and not matched_frag and not overlap_error and not match_number:  # TODO
    #     raise DesignError('%s: Missing required information for writing!' % write_frag_match_info_file.__name__)

    with open(os.path.join(out_path, frag_text_file), 'a+') as out_info_file:
        # if is_initial_match:
        if match_number == 1:
            out_info_file.write('DOCKED POSE ID: %s\n\n' % pose_id)
            out_info_file.write('***** ALL FRAGMENT MATCHES *****\n\n')
            # out_info_file.write("***** INITIAL MATCH FROM REPRESENTATIVES OF INITIAL FRAGMENT CLUSTERS *****\n\n")
        cluster_id = 'i{}_j{}_k{}'.format(*ghost_frag.ijk)
        out_info_file.write(f'MATCH {match_number}\n')
        out_info_file.write(f'z-val: {overlap_error}\n')
        out_info_file.write('CENTRAL RESIDUES\noligomer1 ch, resnum: {}, {}\noligomer2 ch, resnum: {}, {}\n'.format(
            *ghost_frag.get_aligned_chain_and_residue, *matched_frag.get_aligned_chain_and_residue))
        # Todo
        #  out_info_file.write('oligomer1 ch, resnum: %s, %d\n' % (ghost_residue.chain, ghost_residue.residue))
        #  out_info_file.write('oligomer2 ch, resnum: %s, %d\n' % (matched_residue.chain, matched_residue.residue))
        out_info_file.write('FRAGMENT CLUSTER\n')
        out_info_file.write('id: %s\n' % cluster_id)
        out_info_file.write('mean rmsd: %f\n' % ghost_frag.rmsd)
        out_info_file.write('aligned rep: int_frag_%s_%d.pdb\n' % (cluster_id, match_number))
        out_info_file.write('central res pair freqs:\n%s\n\n' % str(central_frequencies))

        # if is_initial_match:
        #     out_info_file.write("***** ALL MATCH(ES) FROM REPRESENTATIVES OF ALL FRAGMENT CLUSTERS *****\n\n")


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
    _heavy_indices: list[int]
    _helix_cb_indices: list[int]
    _side_chain_indices: list[int]
    _contact_order: np.ndarry
    _coords_indexed_residues: np.ndarray  # list[Residue]
    _coords_indexed_residue_atoms: np.ndarray  # list[int]
    _residues: Residues | None
    _residue_indices: list[int] | None
    biomt: list
    biomt_header: str
    file_path: AnyStr | None
    name: str
    secondary_structure: str | None
    sasa: float | None
    structure_containers: list | list[str]
    state_attributes: set[str] = ContainsAtomsMixin.state_attributes | {'_sequence', '_helix_cb_indices'}
    available_letters: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # '0123456789~!@#$%^&*()-+={}[]|:;<>?'

    def __init__(self, atoms: list[Atom] | Atoms = None, residues: list[Residue] | Residues = None,
                 residue_indices: list[int] = None, name: str = None,
                 file_path: AnyStr = None,
                 biomt: list = None, biomt_header: str = None,
                 **kwargs):
        # kwargs passed to StructureBase
        #          parent: StructureBase = None, log: Log | Logger | bool = True, coords: list[list[float]] = None
        super().__init__(**kwargs)
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
        elif atoms:  # is not None
            self._assign_atoms(atoms)
            self._create_residues()
            self._set_coords_indexed()
        else:  # set up an empty Structure or let subclass handle population
            pass

    @classmethod
    def from_file(cls, file: AnyStr, **kwargs):
        """Create a new Model from a file with Atom records"""
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
        """Create a new Model from a .pdb formatted file"""
        return cls(file_path=file, **read_pdb_file(file, **kwargs))

    @classmethod
    def from_mmcif(cls, file: AnyStr, **kwargs):
        """Create a new Model from a .cif formatted file"""
        raise NotImplementedError(mmcif_error)
        return cls(file_path=file, **read_mmcif_file(file, **kwargs))

    @classmethod
    def from_atoms(cls, atoms: list[Atom] | Atoms = None, coords: Coords | np.ndarray = None, **kwargs):
        assert coords, 'Can\'t initialize Structure with Atom objects when no Coords object is passed!'
        return cls(atoms=atoms, coords=coords, **kwargs)

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

    def get_structure_containers(self) -> dict[str, Any]:
        """Return the instance structural containers as a dictionary with attribute as key and container as value"""
        return dict(log=self._log, coords=self._coords, atoms=self._atoms, residues=self._residues)

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
                ''.join([protein_letters_3to1_extended.get(res.type.title(), '-') for res in self.residues])
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

    # @property
    # def coords(self) -> np.ndarray:
    #     """Return the atomic coordinates for the Atoms in the Structure"""
    #     return self._coords.coords[self._atom_indices]

    # @coords.setter
    # def coords(self, coords: Coords | np.ndarray | list[list[float]]):
    #     """Replace the Structure, Atom, and Residue coordinates with specified Coords Object or numpy.ndarray"""
    #     try:
    #         coords.coords
    #     except AttributeError:  # not yet a Coords object, so create one
    #         coords = Coords(coords)
    #     self._coords = coords
    #
    #     if self._coords.coords.shape[0] != 0:
    #         assert len(self.atoms) <= len(self.coords), \
    #             f'{self.name}: ERROR number of Atoms ({len(self.atoms)}) > number of Coords ({len(self.coords)})!'

    # def set_coords(self, coords: Coords | np.ndarray | list[list[float]] = None):  # Todo Depreciate
    #     """Set the coordinates for the Structure as a Coord object. Additionally, updates all member Residues with the
    #     Coords object and maps the atom/coordinate index to each Residue, residue atom index pair.
    #
    #     Only use set_coords once per Structure object creation otherwise Structures with multiple containers will be
    #     corrupted
    #
    #     Args:
    #         coords: The coordinates to set for the structure
    #     """
    #     # self.coords = coords
    #     try:
    #         coords = coords.coords  # if Coords object, extract array
    #     except AttributeError:  # not yet a Coords object, either a np.ndarray or a array like list
    #         pass
    #     self._coords.set(coords)
    #     # self.set_residues_attributes(coords=self._coords)
    #     # self._residues.set_attributes(coords=self._coords)
    #
    #     # index the coordinates to the Residue they belong to and their associated atom_index
    #     residues_atom_idx = [(residue, res_atom_idx) for residue in self.residues for res_atom_idx in residue.range]
    #     self.coords_indexed_residues, self.coords_indexed_residue_atoms = zip(*residues_atom_idx)
    #     # # for every Residue in the Structure set the Residue instance indexed, Atom indices
    #     # range_idx = prior_range_idx = 0
    #     # residue_indexed_ranges = []
    #     # for residue in self.residues:
    #     #     range_idx += residue.number_of_atoms
    #     #     residue_indexed_ranges.append(list(range(prior_range_idx, range_idx)))
    #     #     prior_range_idx = range_idx
    #     # self.residue_indexed_atom_indices = residue_indexed_ranges

    # @property
    # def atom_indices(self) -> list[int] | None:
    #     """The indices which belong to the Structure Atoms/Coords container"""
    #     try:
    #         return self._atom_indices
    #     except AttributeError:
    #         return

    # @atom_indices.setter
    # def atom_indices(self, indices: list[int]):
    #     self._atom_indices = indices

    def _start_indices(self, at: int = 0, dtype: atom_or_residue = None):
        """Modify Structure container indices by a set integer amount

        Args:
            at: The index to insert indices at
            dtype: The type of indices to modify. Can be either 'atom' or 'residue'
        """
        try:  # To get the indices through the public property
            indices = getattr(self, f'{dtype}_indices')
        except AttributeError:
            raise AttributeError(f'The dtype {dtype}_indices was not found the Structure object. Possible values of '
                                 f'dtype are "atom" or "residue"')
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

    @residues.setter
    def residues(self, residues: Residues | list[Residue]):
        """Set the Structure atoms to a Residues object"""
        # Todo make this setter function in the same way as self._coords.replace?
        if isinstance(residues, Residues):
            self._residues = residues
        else:
            self._residues = Residues(residues)

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
        self._assign_atoms(atoms, atoms_only=False)  # no passing of kwargs as below _populate_coords() handles
        # done below with _residues.reindex_atoms(), not necessary here
        # if not self.file_path:  # assume this instance wasn't parsed and Atom indices are incorrect
        #     self._atoms.reindex()

        # set proper residues attributes
        self._residue_indices = list(range(len(residues)))
        if not isinstance(residues, Residues):  # must create the residues object
            residues = Residues(residues)

        if residues.are_dependents():  # copy Residues object to set new attributes on each member Residue
            residues = copy(residues)
            residues.reset_state()  # clear runtime attributes
        self._residues = residues

        self._populate_coords(from_source='residues', **kwargs)  # coords may be passed
        # ensure that coordinates lengths match
        self._validate_coords()
        # update Atom instance attributes to ensure they are dependants of this instance
        self._atoms.set_attributes(_parent=self)
        # update Residue instance attributes to ensure they are dependants of this instance
        # perform after populate_coords due to possible coords setting to ensure that 'residues' .coords not overwritten
        self._residues.set_attributes(_parent=self)
        self._residues.reindex_atoms()
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

    @property
    def ca_coords(self) -> np.ndarray:  # NOT in Residue, ca_coord
        """Return a view of the Coords from the Structure with CA atom coordinates"""
        return self._coords.coords[self.ca_indices]

    @property
    def cb_coords(self) -> np.ndarray:  # NOT in Residue, cb_coord
        """Return a view of the Coords from the Structure with CB atom coordinates"""
        return self._coords.coords[self.cb_indices]

    def get_coords_subset(self, res_start: int, res_end: int, ca: bool = True) -> np.ndarray:
        """Return a view of a subset of the Coords from the Structure specified by a range of Residue numbers"""
        out_coords = []
        if ca:
            for residue in self.get_residues(range(res_start, res_end + 1)):
                out_coords.append(residue.ca_coords)
        else:
            for residue in self.get_residues(range(res_start, res_end + 1)):
                out_coords.extend(residue.coords)

        return np.concatenate(out_coords)

    def _update_structure_container_attributes(self, **kwargs):
        """Update attributes specified by keyword args for all Structure container members"""
        for structure_type in self.structure_containers:
            for structure in getattr(self, structure_type):
                for kwarg, value in kwargs.items():
                    setattr(structure, kwarg, value)

    def set_atoms_attributes(self, **kwargs):
        """Set attributes specified by key, value pairs for Atoms in the Structure

        Keyword Args:
            numbers=None (Container[int]): The Atom numbers of interest
            pdb=False (bool): Whether to search for numbers as they were parsed (True)
        """
        for atom in self.get_atoms(**kwargs):
            for kwarg, value in kwargs.items():
                setattr(atom, kwarg, value)

    def set_residues_attributes(self, **kwargs):
        """Set attributes specified by key, value pairs for Residues in the Structure

        Keyword Args:
            numbers=None (Container[int]): The Residue numbers of interest
            pdb=False (bool): Whether to search for numbers as they were parsed (True)
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

    # @staticmethod
    # def set_structure_attributes(structure, **kwargs):
    #     """Set structure attributes specified by key, value pairs for all object instances in the structure iterator"""
    #     for obj in structure:
    #         for kwarg, value in kwargs.items():
    #             setattr(obj, kwarg, value)

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
            # self._cb_indices = [residue.cb_atom_index for residue in self.residues if residue.cb_atom_index]
            self._cb_indices = [residue.cb_atom_index for residue in self.residues]
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

    def renumber_structure(self):
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
        """Renumber Residue objects sequentially starting with at

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

    # def set_residues(self, residues: list[Residue] | Residues):  # UNUSED
    #     """Set the Structure .residues, ._atom_indices, and .atoms"""
    #     self.residues = residues
    #     self._atom_indices = [idx for residue in self.residues for idx in residue.atom_indices]
    #     self.atoms = self.residues[0]._atoms

    # update_structure():
    #  self._offset_indices() -> self.coords = np.append(self.coords, [atom.coords for atom in atoms]) ->
    #  self.set_atom_coordinates(self.coords) -> self._create_residues() -> self.set_length()

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
                if protein_backbone_atom_types.difference(found_types):  # not an empty set, remove start idx to idx indices
                    remove_atom_indices.extend(list(range(start_atom_index, idx)))
                else:  # proper format
                    # new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx)), atoms=self._atoms,
                    #                             coords=self._coords, log=self._log))
                    new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx)), parent=self))
                start_atom_index = idx
                found_types = {atom.type}  # atom_indices = [idx]
                current_residue_number = atom.residue_number

        # ensure last residue is added after stop iteration
        if protein_backbone_atom_types.difference(found_types):  # not an empty set, remove indices from start idx to idx
            remove_atom_indices.extend(list(range(start_atom_index, idx + 1)))
        else:  # proper format. For each need to increment one higher than the last v
            # new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx + 1)), atoms=self._atoms,
            #                             coords=self._coords, log=self._log))
            new_residues.append(Residue(atom_indices=list(range(start_atom_index, idx + 1)), parent=self))

        self._residue_indices = list(range(len(new_residues)))
        self._residues = Residues(new_residues)

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

    def mutate_residue(self, residue: Residue = None, number: int = None, to: str = 'ALA', **kwargs) -> \
            list[int] | list:
        """Mutate a specific Residue to a new residue type. Type can be 1 or 3 letter format

        Args:
            residue: A Residue object to mutate
            number: A Residue number to select the Residue of interest with
            to: The type of amino acid to mutate to
        Keyword Args:
            pdb: bool = False - Whether to pull the Residue by PDB number
        Returns:
            The indices of the Atoms being removed from the Structure
        """
        # Todo using AA reference, align the backbone + CB atoms of the residue then insert side chain atoms?
        to = protein_letters_1to3.get(to.upper(), to).upper()

        if number is not None:
            residue = self.residue(number, **kwargs)

        if residue is None:
            raise DesignError(f'Cannot {self.mutate_residue.__name__} without passing Residue instance or number')

        residue.type = to
        for atom in residue.atoms:
            atom.residue_type = to

        # Find the corresponding Residue Atom indices to delete. Currently, using side-chain and letting Rosetta handle
        delete_indices = residue.side_chain_indices
        if not delete_indices:  # there are no indices
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
        self._residues.reindex_atoms(start_at=residue.start_index)

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
            raise DesignError(f'This Structure "{self.name}" is not the owner of it\'s attributes and therefore cannot '
                              'handle residue insertion!')
        # Convert incoming aa to residue index so that AAReference can fetch the correct amino acid
        reference_index = \
            protein_letters.find(protein_letters_3to1_extended.get(residue_type.title(), residue_type.upper()))
        if reference_index == -1:
            raise IndexError(f'{self.insert_residue_type.__name__} of residue_type "{residue_type}" is not allowed')
        if at < 1:
            raise IndexError(f'{self.insert_residue_type.__name__} at index {at} < 1 is not allowed')

        # Grab the reference atom coordinates and push into the atom list
        new_residue = copy(reference_aa.residue(reference_index))
        new_residue.number = at
        residue_index = at - 1  # since at is one-indexed integer, take from pose numbering to zero-indexed
        # insert the new_residue coords and atoms into the Structure Atoms
        self._coords.insert(new_residue.coords, at=new_residue.start_index)
        self._atoms.insert(new_residue.atoms, at=new_residue.start_index)
        self._atoms.reindex(start_at=new_residue.start_index)
        # insert the new_residue into the Structure Residues
        self._residues.insert([new_residue], at=residue_index)
        self._residues.reindex_atoms(start_at=residue_index)
        # after coords, atoms, residues insertion into "_" containers, set parent to self
        new_residue.parent = self
        # set new residue_indices and atom_indices
        self._insert_indices(at=residue_index, new_indices=[residue_index], dtype='residue')
        self._insert_indices(at=new_residue.start_index, new_indices=new_residue.atom_indices, dtype='atom')
        # self._atom_indices = self._atom_indices.insert(new_residue.start_index, idx + new_residue.start_index)
        self.renumber_structure()

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
                raise DesignError(f'Can\'t insert_residue_type for an empty {type(self).__name__} class')
            next_residue = None

        # set the new chain_id, number_pdb. Must occur after self._residue_indices update if chain isn't provided
        chain_assignment_error = 'Can\'t solve for the new Residue polymer association automatically! If the new ' \
                                 'Residue is at a Structure termini in a multi-Structure Structure container, you must'\
                                 ' specify which Structure it belongs to by passing chain='
        if chain:
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
    #     return ''.join([protein_letters_3to1_extended.get(res.type.title(), '-') for res in self.residues])

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

    def return_transformed_copy(self, rotation: list[list[float]] | np.ndarray = None,
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
        measure_function: Callable[[Atom], bool]
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
                    bb_info = '\n\t'.join(f'Residue {residue.number:5d}: {atom.return_atom_record()}'
                                          for residue, atom in measured_clashes)
                    self.log.error(f'{self.name} contains {len(measured_clashes)} {measure} clashes from the following '
                                   f'Residues to the corresponding Atom:\n\t{bb_info}')
                raise ClashError(clash_msg)
            else:
                if other_clashes:
                    sc_info = '\n\t'.join(f'Residue {residue.number:5d}: {atom.return_atom_record()}'
                                          for residue, atom in other_clashes)
                    self.log.warning(f'{self.name} contains {len(other_clashes)} {other} clashes between the '
                                     f'following Residues:\n\t{sc_info}')
                return False
        except ClashError as error:  # This was raised from any_clashes()
            if silence_exceptions:
                return True
            else:
                raise error

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
        p = subprocess.Popen([free_sasa_exe_path, f'--format={out_format}', '--probe-radius', str(probe_radius),
                              '-c', free_sasa_configuration_path, '--n-threads=2'] + include_hydrogen,
                             stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(input=self.return_atom_record().encode('utf-8'))
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
        # out, err = p.communicate(input=self.return_atom_record().encode('utf-8'))
        # logger.info(self.return_atom_record()[:120])
        iteration = 1
        all_residue_scores = []
        while iteration < 5:
            p = subprocess.run(errat_cmd, input=self.return_atom_record(), encoding='utf-8', capture_output=True)
            all_residue_scores = p.stdout.strip().split('\n')
            if len(all_residue_scores) - 1 == self.number_of_residues:  # subtract overall_score from all_residue_scores
                break
            iteration += 1

        if iteration == 5:
            self.log.error(f'{self.errat.__name__} couldn\'t generate the correct output length. '
                           f'({len(all_residue_scores) - 1}) != number_of_residues ({self.number_of_residues})')
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
        except (IndexError, AttributeError):
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
        term_window = ''.join(residue.secondary_structure for residue in residues[:window * 2])
        if 'H' * window in term_window:
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

    def return_atom_record(self, **kwargs) -> str:
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
                % '\n'.join('REMARK 350   BIOMT{:1d}{:4d}{:10.6f}{:10.6f}{:10.6f}{:15.5f}'.format(v_idx, m_idx, *vec)
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
    #         file_handle.write(f'{self.return_atom_record(**kwargs)}\n')
    #         return None
    #     else:  # out_path always has default argument current working directory
    #         with open(out_path, 'w') as outfile:
    #             self.write_header(outfile, **kwargs)
    #             outfile.write(f'{self.return_atom_record(**kwargs)}\n')
    #         return out_path

    def get_fragments(self, residues: list[Residue] = None, residue_numbers: list[int] = None, fragment_length: int = 5,
                      **kwargs) -> list[MonoFragment]:
        """From the Structure, find Residues with a matching fragment type as identified in a fragment library

        Args:
            residues: The specific Residues to search for
            residue_numbers: The specific residue numbers to search for
            fragment_length: The length of the fragment observations used
        Keyword Args:
            fragment_db: (FragmentDatabase) = None - The FragmentDatabase with representative fragment types
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
        fragment_range = range(*parameterize_frag_length(fragment_length))
        fragments = []
        for residue_number in residue_numbers:
            frag_residues = self.get_residues(numbers=[residue_number + i for i in fragment_range])

            if len(frag_residues) == fragment_length:
                fragment = MonoFragment(residues=frag_residues, fragment_length=fragment_length, **kwargs)
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
                representatives: dict[int, np.ndarray] = fragment_db.reps
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
            for fragment_type, cluster_coords in representatives.items():
                rmsd, rot, tx = superposition3d(residue_ca_coord_set, cluster_coords)
                if rmsd <= rmsd_thresh and rmsd <= min_rmsd:
                    residue.frag_type = fragment_type
                    min_rmsd = rmsd
                    # min_rmsd, residue.rotation, residue.translation = rmsd, rot, tx

            if residue.frag_type:
                # residue.guide_coords = \
                #     np.matmul(Fragment.template_coords, np.transpose(residue.rotation)) + residue.translation
                residue.fragment_db = fragment_db
                # residue._fragment_coords = residue_ca_coord_set
                residue._representative_coords = representatives[residue.frag_type]
                found_fragments.append(residue)

        return found_fragments

    @property
    def contact_order(self) -> np.ndarray:
        """Return the contact order on a per Residue basis

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
            raise ValueError(f'Can\'t set {self.contact_order.__name__} with a Sequence of length, {len(contact_order)}'
                             f'!= to the number of residues, {self.number_of_residues}')
        for residue, contact_order in zip(residues, contact_order):
            residue.contact_order = contact_order

    # distance of 6 angstroms between heavy atoms was used for 1998 contact order work,
    # subsequent residue wise contact order has focused on the Cb Cb heuristic of 12 A
    # KM thinks that an atom-atom based measure is more accurate, see below for alternative method
    # The BallTree creation is the biggest time cost regardless
    def contact_order_per_residue(self, sequence_distance_cutoff: int = 2, distance: float = 6.) -> list[float]:
        """Calculate the contact order on a per residue basis using calculated heavy atom contacts

        Args:
            sequence_distance_cutoff: The residue spacing required to count a contact as a true contact
            distance: The distance in angstroms to measure atomic contact distances in contact
        Returns:
            The floats representing the contact order for each Residue in the Structure
        """
        # Get heavy Atom coordinates
        coords = self.heavy_coords
        # make and query a tree
        tree = BallTree(coords)
        query = tree.query_radius(coords, distance)

        residues = self.residues
        # in case this was already called, we should set all to 0.
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
    #     """Calculate the contact order on a per residue basis using CB - CB contacts
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
                allowed_aas = {protein_letters_3to1_extended[aa.title()] for aa in allowed_aas}
                allowed_aas = allowed_aas.union(include.get(residue_number, {}))
                res_file_lines.append('%d %s PIKAA %s' % (residue.number, residue.chain, ''.join(sorted(allowed_aas))))
                # res_file_lines.append('%d %s %s' % (residue.number, residue.chain,
                #                                     'PIKAA %s' % ''.join(sorted(allowed_aas)) if len(allowed_aas) > 1
                #                                     else 'NATAA'))
        else:
            for residue, directive in residue_directives.items():
                allowed_aas = residue. \
                    mutation_possibilities_from_directive(directive, background=background.get(residue), **kwargs)
                allowed_aas = {protein_letters_3to1_extended[aa.title()] for aa in allowed_aas}
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

    def __copy__(self):  # -> Self Todo python3.11
        other = self.__class__.__new__(self.__class__)
        # Copy each of the key value pairs to the new, other dictionary
        for attr, obj in self.__dict__.items():
            if attr in parent_attributes:
                # Perform shallow copy on these attributes. They will be handled correctly below
                other.__dict__[attr] = obj
            else:  # Perform a deep copy
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
                self.log.debug(f'The copied {type(self).__name__} is being set as a parent. It was a dependent '
                               f'previously')
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
    # def return_transformed_copy(self, **kwargs):  # rotation=None, translation=None, rotation2=None, translation2=None):
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
    #     new_structures.__init__([structure.return_transformed_copy(**kwargs) for structure in self])
    #     # print('Transformed Structures, structures %s' % [structure for structure in new_structures.structures])
    #     # print('Transformed Structures, models %s' % [structure for structure in new_structures.models])
    #     return new_structures
    #     # return Structures(structures=[structure.return_transformed_copy(**kwargs) for structure in self.structures])

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
    #         file_handle.write('%s\n' % self.return_atom_record(**kwargs))
    #         return
    #
    #     with open(out_path, 'w') as f:
    #         if header:
    #             if isinstance(header, str):
    #                 f.write(header)
    #             # if isinstance(header, Iterable):
    #
    #         if increment_chains:
    #             available_chain_ids = self.chain_id_generator()
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


class ContainsChainsMixin:
    residues: list[Residue]
    chain_ids: list[str]
    chains: list[Chain] | Structures
    log: Logger
    original_chain_ids: list[str]

    def __init__(self, **kwargs):
        # Make dependent on the same list as default. If parsing is done, explicitly set self.original_chain_ids
        self.original_chain_ids = self.chain_ids = []
        super().__init__(**kwargs)

    def _create_chains(self, as_mate: bool = False):
        """For all the Residues in the Structure, create Chain objects which contain their member Residues

        Args:
            as_mate: Whether the Chain instances should be controlled by a captain (True), or dependents of their parent
        Sets:
            self.chain_ids (list[str])
            self.chains (list[Chain] | Structures)
            self.original_chain_ids (list[str])
        """
        residues = self.residues
        residue_idx_start, idx = 0, 1
        prior_residue = residues[0]
        chain_residues = []
        for idx, residue in enumerate(residues[1:], 1):  # start at the second index to avoid off by one
            if residue.number <= prior_residue.number or residue.chain != prior_residue.chain:
                # less than or equal number should only happen with new chain. this SHOULD satisfy a malformed PDB
                chain_residues.append(list(range(residue_idx_start, idx)))
                residue_idx_start = idx
            prior_residue = residue

        # perform after iteration which is the final chain
        chain_residues.append(list(range(residue_idx_start, idx + 1)))  # have to increment as if next residue

        self.chain_ids = remove_duplicates([residue.chain for residue in residues])
        # if self.multimodel:
        self.original_chain_ids = [residues[residue_indices[0]].chain for residue_indices in chain_residues]
        #     self.log.debug(f'Multimodel file found. Original Chains: {",".join(self.original_chain_ids)}')
        # else:
        #     self.original_chain_ids = self.chain_ids

        number_of_chain_ids = len(self.chain_ids)
        if len(chain_residues) != number_of_chain_ids:  # would be different if a multimodel or some weird naming
            available_chain_ids = self.chain_id_generator()
            new_chain_ids = []
            for chain_idx in range(len(chain_residues)):
                if chain_idx < number_of_chain_ids:  # use the chain_ids version
                    chain_id = self.chain_ids[chain_idx]
                else:
                    # chose next available chain unless already taken, then try another
                    chain_id = next(available_chain_ids)
                    while chain_id in self.chain_ids:
                        chain_id = next(available_chain_ids)
                new_chain_ids.append(chain_id)

            self.chain_ids = new_chain_ids

        for residue_indices, chain_id in zip(chain_residues, self.chain_ids):
            self.chains.append(Chain(residue_indices=residue_indices, chain_id=chain_id, as_mate=as_mate, parent=self))

    @property
    def number_of_chains(self) -> int:
        """Return the number of Chain instances in the Structure"""
        return len(self.chains)

    def rename_chains(self, exclude_chains: Sequence = None):
        """Renames chains using Structure.available_letters

        Args:
            exclude_chains: The chains which shouldn't be modified
        Sets:
            self.chain_ids (list[str])
        """
        available_chain_ids = self.chain_id_generator()
        if exclude_chains is None:
            exclude_chains = []

        # Update chain_ids, then each chain
        self.chain_ids = []
        for idx in range(self.number_of_chains):
            chain_id = next(available_chain_ids)
            while chain_id in exclude_chains:
                chain_id = next(available_chain_ids)
            self.chain_ids.append(chain_id)

        for chain, new_id in zip(self.chains, self.chain_ids):
            chain.chain_id = new_id

    def renumber_residues_by_chain(self):
        """For each Chain instance, renumber Residue objects sequentially starting with 1"""
        for chain in self.chains:
            chain.renumber_residues()

    def chain(self, chain_id: str) -> Chain | None:
        """Return the Chain object specified by the passed chain ID from the PDB object

        Args:
            chain_id: The name of the Chain to query
        Returns:
            The Chain if one was found
        """
        for idx, id_ in enumerate(self.chain_ids):
            if id_ == chain_id:
                try:
                    return self.chains[idx]
                except IndexError:
                    raise IndexError(f'The number of chains ({len(self.chains)}) in the {type(self).__name__} != '
                                     f'number of chain_ids ({len(self.chain_ids)})')
        return None

    @staticmethod
    def chain_id_generator() -> Generator[str, None, None]:
        """Provide a generator which produces all combinations of chain ID strings

        Returns
            The generator producing a maximum 2 character string where single characters are exhausted,
                first in uppercase, then in lowercase
        """
        return (first + second for modification in ['upper', 'lower']
                for first in [''] + list(getattr(Structure.available_letters, modification)())
                for second in list(getattr(Structure.available_letters, 'upper')()) +
                list(getattr(Structure.available_letters, 'lower')()))


class Chain(Structure):
    """Create a connected polymer. Usually a subset of the coords and Atom and Residue instances of a larger Structure

    Args:
        as_mate: Whether the Chain instances should be controlled by a captain (True), or dependents of their parent
    """
    _chain_id: str
    _reference_sequence: str

    def __init__(self, chain_id: str = None, name: str = None, as_mate: bool = False, **kwargs):
        super().__init__(name=name if name else chain_id, **kwargs)
        # only if this instance is a Chain, set residues_attributes as in chain_id.setter
        if type(self) == Chain and chain_id is not None:
            self.set_residues_attributes(chain=chain_id)

        if as_mate:
            self.detach_from_parent()

    @property
    def chain_id(self) -> str:
        """The Chain ID for the instance"""
        try:
            return self._chain_id
        except AttributeError:
            self._chain_id = self.residues[0].chain
            return self._chain_id

    @chain_id.setter
    def chain_id(self, chain_id: str):
        self.set_residues_attributes(chain=chain_id)
        self._chain_id = chain_id

    @property
    def reference_sequence(self) -> str:
        """Return the entire Chain sequence, constituting all Residues, not just structurally modelled ones

        Returns:
            The sequence according to the Chain reference, or the Structure sequence if no reference available
        """
        try:
            return self._reference_sequence
        except AttributeError:
            self.log.info('The reference sequence could not be found. Using the observed Residue sequence instead')
            self._reference_sequence = self.sequence
            return self._reference_sequence

    # @reference_sequence.setter
    # def reference_sequence(self, sequence):
    #     self._reference_sequence = sequence


class Entity(SequenceProfile, Chain, ContainsChainsMixin):
    """Entity

    Args:
        chains: A list of Chain instance that match the Entity
        dbref: The unique database reference for the Entity
        reference_sequence: The reference sequence (according to expression sequence or reference database)
    Keyword Args:
        name: str = None - The EntityID. Typically, EntryID_EntityInteger is used to match PDB API identifier format
    """
    _chain_transforms: list[transformation_mapping]
    """The specific transformation operators to generate all mate chains of the Oligomer"""
    _captain: Entity | None
    _chains: list | list[Entity]
    _disorder: dict[int, dict[str, str]]
    _oligomer: Structures  # list[Entity]
    _is_captain: bool
    _is_oligomeric: bool
    _number_of_symmetry_mates: int
    _uniprot_id: str | None
    api_entry: dict[str, dict[str, str]] | None
    dihedral_chain: str | None
    max_symmetry: int | None
    max_symmetry_chain: str | None
    rotation_d: dict[str, dict[str, int | np.ndarray]] | None
    symmetry: str | None

    def __init__(self, chains: list[Chain] | Structures = None, dbref: dict[str, str] = None,
                 reference_sequence: str = None, thermophilic: bool = None, **kwargs):
        """When init occurs chain_ids are set if chains were passed. If not, then they are auto generated"""
        self.thermophilic = thermophilic
        self._chain_transforms = []
        self._captain = None
        self._is_captain = True
        self.api_entry = None  # {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
        self.dihedral_chain = None
        self.max_symmetry = None
        self.max_symmetry_chain = None
        self.rotation_d = {}  # maps mate entities to their rotation matrix
        if chains:  # Instance was initialized with .from_chains()
            # Todo choose most symmetrically average
            #  Move chain symmetry ops below to here?
            representative = chains[0]
            residue_indices = representative.residue_indices
        else:  # Initialized with Structure constructor methods, handle using .is_parent() below
            residue_indices = None

        super().__init__(residue_indices=residue_indices, **kwargs)
        if self.is_parent():  # Todo this logic is not correct. Could be .from_chains() without passing parent!
            self._chains = []
            self._create_chains(as_mate=True)
            chains = self.chains
            # Todo choose most symmetrically average chain
            representative = chains[0]
            # residue_indices = representative.residue_indices
            # # Perform init again with newly parsed chains to set the representative attr as this instances attr
            # super().__init__(residue_indices=residue_indices, parent=representative, **kwargs)
            self._coords.set(representative.coords)
            self._assign_residues(representative.residues, atoms=representative.atoms)
        else:
            # By using extend, we set self.original_chain_ids too
            self.chain_ids.extend([chain.chain_id for chain in chains])

        self._chains = [self]
        # _copy_structure_containers and _update_structure_container_attributes are Entity specific
        self.structure_containers.extend(['_chains'])  # use _chains as chains is okay to equal []
        if len(chains) > 1:
            # Todo handle chains with imperfect symmetry by using the actual chain and forgoing the transform
            #  Need to make a copy of the chain and make it an "Entity mate"
            self._is_oligomeric = True  # inherent in Entity type is a single sequence. Therefore, must be oligomeric
            number_of_residues = self.number_of_residues
            self_seq = self.sequence
            for idx, chain in enumerate(chains[1:]):  # Todo match this mechanism with the symmetric chain index
                chain_seq = chain.sequence
                if chain.number_of_residues == number_of_residues and chain_seq == self_seq:
                    # do an apples to apples comparison
                    # length alone is inaccurate if chain is missing first residue and self is missing it's last...
                    _, rot, tx = superposition3d(chain.cb_coords, self.cb_coords)
                else:  # Do an alignment, get selective indices, then follow with superposition
                    self.log.warning(f'Chain {chain.name} and Entity {self.name} require alignment to symmetrize')
                    fixed_indices, moving_indices = get_equivalent_indices(chain_seq, self_seq)
                    _, rot, tx = superposition3d(chain.cb_coords[fixed_indices], self.cb_coords[moving_indices])
                self._chain_transforms.append(dict(rotation=rot, translation=tx))
                # Todo when capable of asymmetric symmetrization
                # self.chains.append(chain)
            self.number_of_symmetry_mates = len(chains)
            self.symmetry = f'D{self.number_of_symmetry_mates/2}' if self.is_dihedral() \
                else f'C{self.number_of_symmetry_mates}'
        else:
            self.symmetry = None
            self._is_oligomeric = False

        # if chain_ids:
        #     self.chain_ids = chain_ids
        if dbref is not None:
            self.uniprot_id = dbref
        if reference_sequence is not None:
            self._reference_sequence = reference_sequence

    # @classmethod  # Todo implemented above, but clean here to mirror Model?
    # def from_file(cls):
    #     return cls()

    @classmethod
    def from_chains(cls, chains: list[Chain] | Structures = None, **kwargs):
        """Initialize an Entity from a set of Chain objects"""
        return cls(chains=chains, **kwargs)

    @StructureBase.coords.setter
    def coords(self, coords: np.ndarray | list[list[float]]):
        """Set the Coords object while propagating changes to symmetric "mate" chains"""
        if self._is_oligomeric and self._is_captain:
            # **This routine handles imperfect symmetry**
            # self.log.debug('Entity captain is modifying coords')
            # must do these before super().coords.fset()
            # Populate .chains (if not already) with current coords and transformation
            current_chains = self.chains
            # Set current .ca_coords as prior_ca_coords
            prior_ca_coords = self.ca_coords
            current_chain_transforms = self._chain_transforms

            # Set coords with new coords
            super(Structure, Structure).coords.fset(self, coords)  # prefer this over below, as mechanism could change
            # self._coords.replace(self._atom_indices, coords)
            try:
                if self.parent.is_symmetric() and not self._parent_is_updating:
                    # Update the parent coords accordingly if a Pose and is symmetric
                    self.parent.coords = self.parent.coords
            except AttributeError:  # No parent
                pass

            # Find the transformation from the old coordinates to the new
            current_ca_coords = self.ca_coords
            _, new_rot, new_tx = superposition3d(current_ca_coords, prior_ca_coords)

            # Remove prior transforms by setting a fresh container
            self._chain_transforms = []
            # Find the transform between the new coords and the current mate chain coords
            for chain, transform in zip(current_chains[1:], current_chain_transforms):
                # self.log.debug(f'Updated transform of mate {chain.chain_id}')
                # In liu of using chain.coords as lengths might be different
                # Transform prior_coords to chain.coords position, then transform using new_rot and new_tx
                # new_chain_coords = \
                #     np.matmul(np.matmul(prior_ca_coords,
                #                         np.transpose(transform['rotation'])) + transform['translation'],
                #               np.transpose(new_rot)) + new_tx
                new_chain_coords = \
                    np.matmul(chain.ca_coords, np.transpose(new_rot)) + new_tx
                # Find the transform from current coords and the new mate chain coords
                _, rot, tx = superposition3d(new_chain_coords, current_ca_coords)
                # Save transform
                self._chain_transforms.append(dict(rotation=rot, translation=tx))
                # Transform existing mate chain
                chain.coords = np.matmul(coords, np.transpose(rot)) + tx
        else:  # Accept the new coords
            super(Structure, Structure).coords.fset(self, coords)  # prefer this over below, as mechanism could change
            # self._coords.replace(self._atom_indices, coords)

    @property
    def uniprot_id(self) -> str | None:
        """The UniProt ID for the Entity used for accessing genomic and homology features"""
        try:
            return self._uniprot_id
        except AttributeError:
            self.api_entry = query_pdb_by(entity_id=self.name)  # {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
            # self.api_entry = _get_entity_info(self.name)  # {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
            for chain, api_data in self.api_entry.items():  # [next(iter(self.api_entry))]
                # print('Retrieving UNP ID for %s\nAPI DATA for chain %s:\n%s' % (self.name, chain, api_data))
                if api_data.get('db') == 'UNP':
                    # set the first found chain. They are likely all the same anyway
                    self._uniprot_id = api_data.get('accession')
            try:
                return self._uniprot_id
            except AttributeError:
                self._uniprot_id = None
                self.log.warning(f'Entity {self.name}: No uniprot_id found')
        return self._uniprot_id

    @uniprot_id.setter
    def uniprot_id(self, dbref: dict[str, str] | str):
        if isinstance(dbref, dict) and dbref.get('db') == 'UNP':  # Todo make 'UNP' or better 'UKB' a global
            self._uniprot_id = dbref['accession']
        else:
            self._uniprot_id = dbref

    # @property
    # def chain_id(self) -> str:
    #     """The Chain name for the Entity instance"""
    #     return self.residues[0].chain

    @Chain.chain_id.setter
    def chain_id(self, chain_id: str):
        # Same as Chain class property
        self.set_residues_attributes(chain=chain_id)
        self._chain_id = chain_id
        # Different from Chain class
        if self._is_captain:
            self._set_chain_ids()

    def _set_chain_ids(self):
        """From the Entity.chain_id set all mate Chains with an incrementally higher id

        Sets:
            self.chain_ids:
                list(str)
        """
        first_chain_id = self.chain_id
        self.chain_ids = [first_chain_id]  # use the existing chain_id
        chain_gen = self.chain_id_generator()
        # Iterate over the generator until the current chain_id is found
        discard = next(chain_gen)
        while discard != first_chain_id:
            discard = next(chain_gen)

        # Iterate over the generator adding each successive chain_id to self.chain_ids
        for idx in range(1, self.number_of_symmetry_mates):  # Only get ids for the mate chains
            chain_id = next(chain_gen)
            # while chain_id in self.chain_ids:
            #     chain_id = next(chain_gen)
            # chain.chain_id = chain_id
            self.chain_ids.append(chain_id)

        # Must set chain_ids first, then chains
        for idx, chain in enumerate(self.chains[1:], 1):
            chain.chain_id = self.chain_ids[idx]

    # @property
    # def chain_ids(self) -> list:  # Also used in Model
    #     """The names of each Chain found in the Entity"""
    #     try:
    #         return self._chain_ids
    #     except AttributeError:  # This shouldn't be possible with the constructor available
    #         available_chain_ids = self.chain_id_generator()
    #         self._chain_ids = [self.chain_id]
    #         for _ in range(self.number_of_symmetry_mates - 1):
    #             next_chain = next(available_chain_ids)
    #             while next_chain in self._chain_ids:
    #                 next_chain = next(available_chain_ids)
    #
    #             self._chain_ids.append(next_chain)
    #
    #         return self._chain_ids
    #
    # @chain_ids.setter
    # def chain_ids(self, chain_ids: list[str]):
    #     self._chain_ids = chain_ids

    @property
    def number_of_symmetry_mates(self) -> int:
        """The number of copies of the Entity in the Oligomer"""
        try:
            return self._number_of_symmetry_mates
        except AttributeError:  # set based on the symmetry, unless that fails then find using chain_ids
            self._number_of_symmetry_mates = valid_subunit_number.get(self.symmetry, len(self.chain_ids))
            return self._number_of_symmetry_mates

    @number_of_symmetry_mates.setter
    def number_of_symmetry_mates(self, number_of_symmetry_mates: int):
        self._number_of_symmetry_mates = number_of_symmetry_mates

    @property
    def center_of_mass_symmetric(self) -> np.ndarray:  # Todo mirrors SymmetricModel
        """The center of mass for the entire symmetric system"""
        # number_of_symmetry_atoms = len(self.symmetric_coords)
        # return np.matmul(np.full(number_of_symmetry_atoms, 1 / number_of_symmetry_atoms), self.symmetric_coords)
        # v since all symmetry by expand_matrix anyway
        return self.center_of_mass_symmetric_models.mean(axis=-2)

    @property
    def center_of_mass_symmetric_models(self) -> np.ndarray:
        """The individual centers of mass for each model in the symmetric system"""
        # try:
        #     return self._center_of_mass_symmetric_models
        # except AttributeError:
        com = self.center_of_mass
        mate_coms = [com]
        for transform in self._chain_transforms:
            mate_coms.append(np.matmul(com, transform['rotation'].T) + transform['translation'])

        print('MATE COMS', mate_coms)
        # np.array makes the right shape while concatenate doesn't
        return np.array(mate_coms)
        # self._center_of_mass_symmetric_models = np.array(mate_coms)
        # return self._center_of_mass_symmetric_models

    def is_captain(self) -> bool:
        """Is the Entity instance the captain?"""
        return self._is_captain

    def is_mate(self) -> bool:
        """Is the Entity instance a mate?"""
        return not self._is_captain

    def is_oligomeric(self) -> bool:
        """Is the Entity oligomeric?"""
        return self._is_oligomeric

    # def _remove_chain_transforms(self):
    #     """Remove _chains and _chain_transforms, set prior_ca_coords in preparation for coordinate movement"""
    #     self._chains.clear()  # useful for mate chains...
    #     self._chain_transforms.clear()
    #     del self._chain_transforms  # useful for mate chains as their transforms aren't correctly oriented
    #     self.prior_ca_coords = self.ca_coords

    @property
    def entities(self) -> list[Entity]:  # Structures
        """Returns an iterator over the Entity"""
        return [self]

    @property
    def number_of_entities(self) -> int:
        """Return the number of Entity instances in the Structure"""
        return 1

    @property
    def chains(self) -> list[Entity]:  # Structures
        """Returns transformed copies of the Entity"""
        # Set in init -> self._chains = [self]
        if len(self._chains) == 1 and self._chain_transforms and self._is_captain:
            # populate ._chains with Entity mates
            self._chains.extend([self.return_transformed_mate(**transform) for transform in self._chain_transforms])
            chain_ids = self.chain_ids
            self.log.debug(f'Entity chains property has {len(self._chains)} chains because the underlying '
                           f'chain_transforms has {len(self._chain_transforms)}. chain_ids has {len(chain_ids)}')
            for idx, chain in enumerate(self._chains[1:], 1):
                # set entity.chain_id which sets all residues
                chain.chain_id = chain_ids[idx]
        # else:
        #     return self._chains

        return self._chains

    @property
    def reference_sequence(self) -> str:
        """Return the entire Entity sequence, constituting all Residues, not just structurally modelled ones

        Returns:
            The sequence according to the Entity reference, or the Structure sequence if no reference available
        """
        try:
            return self._reference_sequence
        except AttributeError:
            self._retrieve_sequence_from_api()
            if not self._reference_sequence:
                self.log.warning('The reference sequence could not be found. Using the observed Residue sequence '
                                 'instead')
                self._reference_sequence = self.sequence
            return self._reference_sequence

    # @reference_sequence.setter
    # def reference_sequence(self, sequence):
    #     self._reference_sequence = sequence

    @property
    def disorder(self) -> dict[int, dict[str, str]]:
        """Return the Residue number keys where disordered residues are found by comparison of the genomic (construct)
        sequence with that of the structure sequence

        Returns:
            Mutation index to mutations in the format of {1: {'from': 'A', 'to': 'K'}, ...}
        """
        try:
            return self._disorder
        except AttributeError:
            self._disorder = generate_mutations(self.reference_sequence, self.sequence, only_gaps=True)
            return self._disorder

    # def chain(self, chain_name: str) -> Entity | None:
    #     """Fetch and return an Entity by chain name"""
    #     for idx, chain_id in enumerate(self.chain_ids):
    #         if chain_id == chain_name:
    #             try:
    #                 return self.chains[idx]
    #             except IndexError:
    #                 raise IndexError(f'The number of chains ({len(self.chains)}) in the {type(self).__name__} != '
    #                                  f'number of chain_ids ({len(self.chain_ids)})')

    def _retrieve_sequence_from_api(self, entity_id: str = None):
        """Using the Entity ID, fetch information from the PDB API and set the instance reference_sequence"""
        if not entity_id:
            # if len(self.name.split('_')) == 2:
            #     entity_id = self.name
            # else:
            try:
                entry, entity_integer, *_ = self.name.split('_')
                if len(entry) == 4 and entity_integer.isdigit():
                    entity_id = f'{entry}_{entity_integer}'
            except ValueError:  # couldn't unpack enough
                self.log.warning(f'{self._retrieve_sequence_from_api.__name__}: If an entity_id isn\'t passed and the '
                                 f'Entity name "{self.name}" is not the correct format (1abc_1), the query will fail. '
                                 f'Retrieving closest entity_id by PDB API structure sequence')
                entity_id = retrieve_entity_id_by_sequence(self.sequence)
                if not entity_id:
                    self._reference_sequence = None
                    return

        self.log.debug(f'Querying {entity_id} reference sequence from PDB')
        self._reference_sequence = get_entity_reference_sequence(entity_id=entity_id)

    # def retrieve_info_from_api(self):
    #     """Retrieve information from the PDB API about the Entity
    #
    #     Sets:
    #         self.api_entry (dict): {chain: {'accession': 'Q96DC8', 'db': 'UNP'}, ...}
    #     """
    #     self.api_entry = get_pdb_info_by_entity(self.name)

    @property
    def oligomer(self) -> list[Entity] | Structures:
        """Access the oligomeric Structure which is a copy of the Entity plus any additional symmetric mate chains

        Returns:
            Structures object with the underlying chains in the oligomer
        """
        try:
            return self._oligomer
        except AttributeError:
            # if not self._is_oligomeric:
            #     self.log.warning('The oligomer was requested but the Entity %s is not oligomeric. Returning the Entity '
            #                      'instead' % self.name)
            # self._oligomer = self.chains  # OLD WAY
            self._oligomer = Structures(self.chains, parent=self)  # NEW WAY
            # self._oligomer = Structures(self.chains)
            return self._oligomer

    def remove_mate_chains(self):
        """Clear the Entity of all Chain and Oligomer information"""
        self._chain_transforms.clear()
        self.number_of_symmetry_mates = 1
        # self._chains.clear()
        self._chains = [self]
        self._is_captain = False
        self.chain_ids = [self.chain_id]

    def _make_mate(self, other: Entity):
        """Turn the Entity into a "mate" Entity"""
        self._captain = other
        self._is_captain = False
        self.chain_ids = [self.chain_id]  # set for a length of 1, using the captain self.chain_id
        self._chain_transforms.clear()

    def _make_captain(self):
        """Turn the Entity into a "captain" Entity if it isn't already"""
        if not self._is_captain:
            # self.log.debug(f'Promoting mate Entity {self.chain_id} to a captain')
            # Todo handle superposition with imperfect symmetry
            # Find and save the transforms between the self.coords and the prior captains mate chains
            current_ca_coords = self.ca_coords
            self._chain_transforms = []
            # for idx, chain in enumerate(self._captain.chains, 1):
            for chain in self._captain.chains:
                # Find the transform from current coords and the new mate chain coords
                _, rot, tx = superposition3d(chain.ca_coords, current_ca_coords)
                if np.allclose(identity_matrix, rot):
                    # This "chain" is the instance of the self, we don't need the identity
                    # self.log.debug(f'Skipping identity transform')
                    continue
                # self.log.debug(f'Adding transform between self._captain.chain idx {idx} and the new captain')
                self._chain_transforms.append(dict(rotation=rot, translation=tx))

            # # Alternative:
            # # Transform the transforms by finding transform from the old captain to the current coords
            # # Not sure about the algebraic requirements of the old translation. It may require rotation with offset_ro...
            # _, offset_rot, offset_tx = superposition3d(self.ca_coords, self._captain.ca_coords)
            # self._chain_transforms = []  # self._captain._chain_transforms.copy()
            # for idx, transform in enumerate(self._captain._chain_transforms):  # self._chain_transforms):
            #     # Rotate the captain oriented rotation matrix to the current coordinates
            #     new_rotation = np.matmul(transform['rotation'], offset_rot)
            #     new_transform = transform['translation'] + offset_tx
            #     self._chain_transforms.append()

            self._is_captain = True
            self.chain_id = self._captain.chain_id
            self._captain = None

    def make_oligomer(self, symmetry: str = None, rotation: list[list[float]] | np.ndarray = None,
                      translation: list[float] | np.ndarray = None, rotation2: list[list[float]] | np.ndarray = None,
                      translation2: list[float] | np.ndarray = None, **kwargs):
        """Given a symmetry and transformational mapping, generate oligomeric copies of the Entity

        Assumes that the symmetric system treats the canonical symmetric axis as the Z-axis, and if the Entity is not at
        the origin, that a transformation describing its current position relative to the origin is passed so that it
        can be moved to the origin. At the origin, makes the required oligomeric rotations, to generate an oligomer
        where symmetric copies are stored in the .chains attribute then reverses the operations back to original
        reference frame if any was provided

        Args:
            symmetry: The symmetry to set the Entity to
            rotation: The first rotation to apply, expected array shape (3, 3)
            translation: The first translation to apply, expected array shape (3,)
            rotation2: The second rotation to apply, expected array shape (3, 3)
            translation2: The second translation to apply, expected array shape (3,)
        Sets:
            self._chain_transforms (list[transformation_mapping])
            self._is_oligomeric=True (bool)
            self.number_of_symmetry_mates (int)
            self.symmetry (str)
        """
        try:
            if symmetry is None or symmetry == 'C1':  # not symmetric
                return
            elif symmetry in cubic_point_groups:
                # must transpose these along last axis as they are pre-transposed upon creation
                rotation_matrices = point_group_symmetry_operators[symmetry].swapaxes(-2, -1)
                degeneracy_matrices = None  # Todo may need to add T degeneracy here!
            elif 'D' in symmetry:  # provide a 180-degree rotation along x (all D orient symmetries have axis here)
                rotation_matrices = get_rot_matrices(rotation_range[symmetry.replace('D', 'C')], 'z', 360)
                degeneracy_matrices = [identity_matrix, flip_x_matrix]
            else:  # symmetry is cyclic
                rotation_matrices = get_rot_matrices(rotation_range[symmetry], 'z')
                degeneracy_matrices = None
            degeneracy_rotation_matrices = make_rotations_degenerate(rotation_matrices, degeneracy_matrices)
        except KeyError:
            raise ValueError(f'The symmetry {symmetry} is not viable! You should try to add compatibility '
                             f'for it if you believe this is a mistake')

        self.symmetry = symmetry
        # self._is_captain = True
        # Todo should this be set here. NO! set in init
        #  or prevent self._mate from becoming oligomer?
        self._is_oligomeric = True
        if rotation is None:
            rotation = inv_rotation = identity_matrix
        else:
            inv_rotation = np.linalg.inv(rotation)
        if translation is None:
            translation = origin

        if rotation2 is None:
            rotation2 = inv_rotation2 = identity_matrix
        else:
            inv_rotation2 = np.linalg.inv(rotation2)
        if translation2 is None:
            translation2 = origin
        # this is helpful for dihedral symmetry as entity must be transformed to origin to get canonical dihedral
        # entity_inv = entity.return_transformed_copy(rotation=inv_expand_matrix, rotation2=inv_set_matrix[group])
        # need to reverse any external transformation to the entity coords so rotation occurs at the origin...
        # and undo symmetry expansion matrices
        # centered_coords = transform_coordinate_sets(self.coords, translation=-translation2,
        # centered_coords = transform_coordinate_sets(self._coords.coords, translation=-translation2)
        cb_coords = self.cb_coords
        centered_coords = transform_coordinate_sets(cb_coords, translation=-translation2)

        centered_coords_inv = transform_coordinate_sets(centered_coords, rotation=inv_rotation2,
                                                        translation=-translation, rotation2=inv_rotation)
        self._chain_transforms.clear()
        number_of_subunits = 0
        for rotation_matrix in degeneracy_rotation_matrices:
            number_of_subunits += 1
            if number_of_subunits == 1 and np.all(rotation_matrix == identity_matrix):
                self.log.debug(f'Skipping {self.make_oligomer.__name__} transformation 1 as it is identity')
                continue
            rot_centered_coords = transform_coordinate_sets(centered_coords_inv, rotation=rotation_matrix)
            new_coords = transform_coordinate_sets(rot_centered_coords, rotation=rotation, translation=translation,
                                                   rotation2=rotation2, translation2=translation2)
            _, rot, tx = superposition3d(new_coords, cb_coords)
            self._chain_transforms.append(dict(rotation=rot, translation=tx))

        # Set the new properties
        self.number_of_symmetry_mates = number_of_subunits
        self._set_chain_ids()

    def return_transformed_mate(self, rotation: list[list[float]] | np.ndarray = None,
                                translation: list[float] | np.ndarray = None,
                                rotation2: list[list[float]] | np.ndarray = None,
                                translation2: list[float] | np.ndarray = None) -> Entity:
        """Make a semi-deep copy of the Entity, stripping any captain attributes, transforming the coordinates

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
            new_coords = np.matmul(self.coords, np.transpose(rotation))
        else:
            new_coords = self.coords

        if translation is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation)

        if rotation2 is not None:  # required for np.ndarray or None checks
            np.matmul(new_coords, np.transpose(rotation2), out=new_coords)

        if translation2 is not None:  # required for np.ndarray or None checks
            new_coords += np.array(translation2)

        new_structure = copy(self)
        new_structure._make_mate(self)
        # _make_mate executes the following
        # self._is_captain = False
        # self._chain_ids = [self.chain_id]
        # self._chain_transforms.clear()
        new_structure.coords = new_coords

        return new_structure

    def format_header(self, **kwargs) -> str:
        """Return the BIOMT and the SEQRES records based on the Entity

        Returns:
            The header with PDB file formatting
        """
        return super().format_header(**kwargs) + self.format_seqres(**kwargs)
        # return super().format_header(**kwargs) + self.format_biomt(**kwargs) + self.format_seqres(**kwargs)

    def format_seqres(self, asu: bool = True, **kwargs) -> str:
        """Format the reference sequence present in the SEQRES remark for writing to the output header

        Args:
            asu: Whether to output the Entity ASU or the full oligomer
        Returns:
            The PDB formatted SEQRES record
        """
        asu_slice = 1 if asu else None  # This is the only difference from Model
        formated_reference_sequence = \
            {chain.chain_id: ' '.join(map(str.upper, (protein_letters_1to3_extended.get(aa, 'XXX')
                                                      for aa in chain.reference_sequence)))
             for chain in self.chains[:asu_slice]}
        chain_lengths = {chain: len(sequence) for chain, sequence in formated_reference_sequence.items()}
        return '%s\n' \
               % '\n'.join(f'SEQRES{line_number:4d} {chain:1s}{chain_lengths[chain]:5d}  '
                           f'{sequence[seq_res_len * (line_number - 1):seq_res_len * line_number]}         '
                           for chain, sequence in formated_reference_sequence.items()
                           for line_number in range(1, 1 + math.ceil(chain_lengths[chain] / seq_res_len)))

    # Todo overwrite Structure.write() method with oligomer=True flag?
    def write_oligomer(self, out_path: bytes | str = os.getcwd(), file_handle: IO = None, header: str = None,
                       **kwargs) -> str | None:
        #               header=None,
        """Write Entity.oligomer Structure to a file specified by out_path or with a passed file_handle

        Args:
            out_path: The location where the Structure object should be written to disk
            file_handle: Used to write Structure details to an open FileObject
            header: A string that is desired at the top of the file
        Keyword Args:
            asu: (bool) = True - Whether to output SEQRES for the Entity ASU or the full oligomer
        Returns:
            The name of the written file if out_path is used
        """
        offset = 0
        if file_handle:
            for chain in self.chains:
                file_handle.write(f'{chain.return_atom_record(atom_offset=offset, **kwargs)}\n')
                offset += chain.number_of_atoms
            return None

        if out_path:
            _header = self.format_header(asu=False, **kwargs)
            if header is not None and isinstance(header, str):  # used for cryst_record now...
                _header += (header if header[-2:] == '\n' else f'{header}\n')

            with open(out_path, 'w') as outfile:
                outfile.write(_header)  # function implies we want all chains, i.e. asu=False
                for chain in self.chains:
                    outfile.write(f'{chain.return_atom_record(atom_offset=offset, **kwargs)}\n')
                    offset += chain.number_of_atoms

            return out_path

    def orient(self, symmetry: str = None, log: AnyStr = None):  # similar function in Model
        """Orient a symmetric PDB at the origin with its symmetry axis canonically set on axes defined by symmetry
        file. Automatically produces files in PDB numbering for proper orient execution

        Args:
            symmetry: What is the symmetry of the specified PDB?
            log: If there is a log specific for orienting
        """
        # orient_oligomer.f program notes
        # C		Will not work in any of the infinite situations where a PDB file is f***ed up,
        # C		in ways such as but not limited to:
        # C     equivalent residues in different chains don't have the same numbering; different subunits
        # C		are all listed with the same chain ID (e.g. with incremental residue numbering) instead
        # C		of separate IDs; multiple conformations are written out for the same subunit structure
        # C		(as in an NMR ensemble), negative residue numbers, etc. etc.
        # must format the input.pdb in an acceptable manner

        try:
            subunit_number = valid_subunit_number[symmetry]
        except KeyError:
            self.log.error(f'{self.orient.__name__}: Symmetry {symmetry} is not a valid symmetry. '
                           f'Please try one of: {", ".join(valid_symmetries)}')
            return

        if not log:
            log = self.log

        if self.file_path:
            file_name = os.path.basename(self.file_path)
        else:
            file_name = f'{self.name}.pdb'
        # Todo change output to logger with potential for file and stdout

        number_of_subunits = self.number_of_chains
        if symmetry == 'C1':
            log.debug('C1 symmetry doesn\'t have a cannonical orientation')
            self.translate(-self.center_of_mass)
            return
        elif number_of_subunits > 1:
            if number_of_subunits != subunit_number:
                raise ValueError(f'{file_name} could not be oriented: It has {number_of_subunits} subunits '
                                 f'while a multiple of {subunit_number} are expected for {symmetry} symmetry')
        else:
            raise ValueError(f'{self.name}: Cannot orient a Structure with only a single chain. No symmetry present!')

        orient_input = Path(orient_dir, 'input.pdb')
        orient_output = Path(orient_dir, 'output.pdb')

        def clean_orient_input_output():
            orient_input.unlink(missing_ok=True)
            orient_output.unlink(missing_ok=True)

        clean_orient_input_output()
        # Have to change residue numbering to PDB numbering
        self.write_oligomer(out_path=str(orient_input), pdb_number=True)

        # Todo superposition3d -> quaternion
        p = subprocess.Popen([orient_exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, cwd=orient_dir)
        in_symm_file = os.path.join(orient_dir, 'symm_files', symmetry)
        stdout, stderr = p.communicate(input=in_symm_file.encode('utf-8'))
        log.info(file_name + stdout.decode()[28:])
        log.info(stderr.decode()) if stderr else None
        if not orient_output.exists() or orient_output.stat().st_size == 0:
            log_file = getattr(log.handlers[0], 'baseFilename', None)
            log_message = f'. Check {log_file} for more information' if log_file else ''
            raise RuntimeError(f'orient_oligomer could not orient {file_name}{log_message}')

        oriented_pdb = Entity.from_file(str(orient_output), name=self.name, log=log)
        orient_fixed_struct = oriented_pdb.chains[0]
        moving_struct = self

        orient_fixed_seq = orient_fixed_struct.sequence
        moving_seq = moving_struct.sequence

        if orient_fixed_struct.number_of_residues == moving_struct.number_of_residues and orient_fixed_seq == moving_seq:
            # do an apples to apples comparison
            # length alone is inaccurate if chain is missing first residue and self is missing it's last...
            _, rot, tx = superposition3d(orient_fixed_struct.cb_coords, moving_struct.cb_coords)
        else:  # do an alignment, get selective indices, then follow with superposition
            self.log.warning(f'{moving_struct.chain_id} and {orient_fixed_struct.chain_id} require alignment to '
                             f'{self.orient.__name__}')
            fixed_indices, moving_indices = get_equivalent_indices(orient_fixed_seq, moving_seq)
            _, rot, tx = superposition3d(orient_fixed_struct.cb_coords[fixed_indices],
                                         moving_struct.cb_coords[moving_indices])

        self.transform(rotation=rot, translation=tx)
        clean_orient_input_output()

    def find_chain_symmetry(self):
        """Search for the chain symmetry by using quaternion geometry to solve the symmetric order of the rotations
         which superimpose chains on the Entity. Translates the Entity to the origin using center of mass, then the axis
        of rotation only needs to be translated to the center of mass to recapitulate the specific symmetry operation

        Requirements - all chains are the same length

        Sets:
            self.rotation_d (dict[Entity, dict[str, int | np.ndarray]])
            self.max_symmetry_chain (str)
        Returns:
            The name of the file written for symmetry definition file creation
        """
        # Find the superposition from the Entity to every mate chain
        center_of_mass = self.center_of_mass
        symmetric_center_of_mass = self.center_of_mass_symmetric
        print('symmetric_center_of_mass', symmetric_center_of_mass)
        cb_coords = self.cb_coords
        for chain in self.chains[1:]:
            # System must be transformed to the origin
            rmsd, quat, tx = superposition3d(cb_coords, chain.cb_coords, quaternion=True)
            # rmsd, quat, tx = superposition3d(cb_coords-center_of_mass, chain.cb_coords-center_of_mass, quaternion=True)
            self.log.debug(f'rmsd={rmsd} quaternion={quat} translation={tx}')
            # python pseudo
            w = abs(quat[3])
            omega = math.acos(w)
            symmetry_order = int(math.pi/omega + .5)  # round to the nearest integer
            self.log.debug(f'{chain.chain_id}:{symmetry_order}-fold axis {quat[0]:8f} {quat[1]:8f} {quat[2]:8f}')
            self.rotation_d[chain.chain_id] = {'sym': symmetry_order, 'axis': quat[:3]}

        # if not struct_file:
        #     struct_file = self.write_oligomer(out_path=f'make_sdf_input-{self.name}-{random() * 100000:.0f}.pdb')
        #
        # This script translates the center of mass to the origin then uses quaternion geometry to solve for the
        # rotations which superimpose chains provided by -i onto a designated chain (usually A). It returns the order of
        # the rotation as well as the axis along which the rotation must take place. The axis of the rotation only needs
        # to be translated to the center of mass to recapitulate the specific symmetry operation.
        #
        #  perl symdesign/dependencies/rosetta/sdf/scout_symmdef_file.pl -p 1ho1_tx_4.pdb -i B C D E F G H
        #  >B:3-fold axis: -0.00800197 -0.01160998 0.99990058
        #  >C:3-fold axis: 0.00000136 -0.00000509 1.00000000
        #
        # start_chain, *rest = self.chain_ids
        # scout_cmd = ['perl', scout_symmdef, '-p', struct_file, '-a', start_chain, '-i'] + rest
        # self.log.debug(f'Scouting chain symmetry: {subprocess.list2cmdline(scout_cmd)}')
        # p = subprocess.Popen(scout_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # out, err = p.communicate()
        # self.log.debug(out.decode('utf-8').strip().split('\n'))
        #
        # for line in out.decode('utf-8').strip().split('\n'):
        #     chain, symmetry, axis = line.split(':')
        #     self.rotation_d[chain] = \
        #         {'sym': int(symmetry[:6].rstrip('-fold')), 'axis': np.array(list(map(float, axis.strip().split())))}
        #     # the returned axis is from a center of mass at the origin as Structure has been translated there
        #
        # return struct_file

        # Find the highest order symmetry in the Structure
        max_sym, max_chain_id = 0, None
        for chain, data in self.rotation_d.items():
            if data['sym'] > max_sym:
                max_sym = data['sym']
                max_chain_id = chain

        self.max_symmetry = max_sym
        self.max_symmetry_chain = max_chain_id

    def is_dihedral(self) -> bool:
        """Report whether a structure is dihedral or not

        Returns:
            True if the Structure is dihedral, False if not
        """
        if self.max_symmetry_chain is None:
            self.find_chain_symmetry()

        # if 1 < self.number_of_symmetry_mates/self.max_symmetry < 2:
        #     self.log.critical(f'{self.name} symmetry is malformed! Highest symmetry ({max_symmetry_data["sym"]}-fold)'
        #                       f' is less than 2x greater than the number ({self.number_of_symmetry_mates}) of chains')

        return self.number_of_symmetry_mates/self.max_symmetry == 2

    def find_dihedral_chain(self) -> Entity | None:
        """From the symmetric system, find a dihedral chain and return the instance

        Sets:
            self.dihedral_chain (str): The name of the chain that is dihedral
        Returns:
            The dihedral mate chain
        """
        if not self.is_dihedral():
            return None

        # Ensure if the structure is dihedral a selected dihedral_chain is orthogonal to the maximum symmetry axis
        max_symmetry_data = self.rotation_d[self.max_symmetry_chain]
        for chain, data in self.rotation_d.items():
            if data['sym'] == 2:
                axis_dot_product = np.dot(max_symmetry_data['axis'], data['axis'])
                if axis_dot_product < 0.01:
                    if np.allclose(data['axis'], [1, 0, 0]):
                        self.log.debug(f'The relation between {self.max_symmetry_chain} and {chain} would result in a '
                                       f'malformed .sdf file')
                        pass  # this will not work in the make_symmdef.pl script, we should choose orthogonal y-axis
                    else:
                        return chain

        return None

    def make_sdf(self, struct_file: AnyStr = None, out_path: AnyStr = os.getcwd(), **kwargs) -> \
            AnyStr:
        """Use the make_symmdef_file.pl script from Rosetta to make a symmetry definition file on the Structure

        perl $ROSETTA/source/src/apps/public/symmetry/make_symmdef_file.pl -p filepath/to/pdb.pdb -i B -q

        Args:
            struct_file: The location of the input .pdb file
            out_path: The location the symmetry definition file should be written
        Keyword Args:
            modify_sym_energy_for_cryst=False (bool): Whether the symmetric energy produced in the file should be modified
            energy=2 (int): Scalar to modify the Rosetta energy by
        Returns:
            Symmetry definition filename
        """
        out_file = os.path.join(out_path, f'{self.name}.sdf')
        if os.path.exists(out_file):
            return out_file

        # if self.symmetry == 'C1':
        #     return
        # el
        if self.symmetry in cubic_point_groups:
            # if not struct_file:
            #     struct_file = self.write_oligomer(out_path='make_sdf_input-%s-%d.pdb' % (self.name, random() * 100000))
            sdf_mode = 'PSEUDO'
            self.log.warning('Using experimental symmetry definition file generation, proceed with caution as Rosetta '
                             'runs may fail due to improper set up')
        else:
            # if not struct_file:
            #     struct_file = self.scout_symmetry(struct_file=struct_file)
            sdf_mode = 'NCS'

        if not struct_file:
            # struct_file = self.scout_symmetry(struct_file=struct_file)
            struct_file = self.write_oligomer(out_path=f'make_sdf_input-{self.name}-{random() * 100000:.0f}.pdb')

        if self.is_dihedral():
            dihedral_chain = self.find_dihedral_chain()
            chains = [self.max_symmetry_chain, dihedral_chain]
        else:
            chains = [self.max_symmetry_chain]

        sdf_cmd = \
            ['perl', make_symmdef, '-m', sdf_mode, '-q', '-p', struct_file, '-a', self.chain_ids[0], '-i'] + chains
        self.log.info(f'Creating symmetry definition file: {subprocess.list2cmdline(sdf_cmd)}')
        # with open(out_file, 'w') as file:
        p = subprocess.Popen(sdf_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, err = p.communicate()

        if os.path.exists(struct_file):
            os.system(f'rm {struct_file}')
        if p.returncode != 0:
            raise DesignError(f'Symmetry definition file creation failed for {self.name}')

        self.format_sdf(out.decode('utf-8').split('\n')[:-1], to_file=out_file, dihedral=dihedral, **kwargs)
        #               modify_sym_energy_for_cryst=False, energy=2)

        return out_file

    def format_sdf(self, lines: list, to_file: AnyStr = None,
                   out_path: AnyStr = os.getcwd(), dihedral: bool = False,
                   modify_sym_energy_for_cryst: bool = False, energy: int = None) -> AnyStr:
        """Ensure proper sdf formatting before proceeding

        Args:
            lines: The symmetry definition file lines
            to_file: The name of the symmetry definition file
            out_path: The location the symmetry definition file should be written
            dihedral: Whether the assembly is in dihedral symmetry
            modify_sym_energy_for_cryst: Whether the symmetric energy should match crystallographic systems
            energy: Scalar to modify the Rosetta energy by
        Returns:
            The location the symmetry definition file was written
        """
        subunits, virtuals, jumps_com, jumps_subunit, trunk = [], [], [], [], []
        for idx, line in enumerate(lines, 1):
            if line.startswith('xyz'):
                virtual = line.split()[1]
                if virtual.endswith('_base'):
                    subunits.append(virtual)
                else:
                    virtuals.append(virtual.lstrip('VRT'))
                # last_vrt = line + 1
            elif line.startswith('connect_virtual'):
                jump = line.split()[1].lstrip('JUMP')
                if jump.endswith('_to_com'):
                    jumps_com.append(jump[:-7])
                elif jump.endswith('_to_subunit'):
                    jumps_subunit.append(jump[:-11])
                else:
                    trunk.append(jump)
                last_jump = idx  # index where the VRTs and connect_virtuals end. The "last jump"

        assert set(trunk) - set(virtuals) == set(), 'Symmetry Definition File VRTS are malformed'
        assert self.number_of_symmetry_mates == len(subunits), 'Symmetry Definition File VRTX_base are malformed'

        if dihedral:  # Remove dihedral connecting (trunk) virtuals: VRT, VRT0, VRT1
            virtuals = [virtual for virtual in virtuals if len(virtual) > 1]  # subunit_
        else:
            if '' in virtuals:
                virtuals.remove('')

        jumps_com_to_add = set(virtuals).difference(jumps_com)
        count = 0
        if jumps_com_to_add != set():
            for count, jump_com in enumerate(jumps_com_to_add, count):
                lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_com VRT%s VRT%s_base'
                             % (jump_com, jump_com, jump_com))
            lines[-2] = lines[-2].strip() + (len(jumps_com_to_add) * ' JUMP%s_to_subunit') % tuple(jumps_com_to_add)

        jumps_subunit_to_add = set(virtuals).difference(jumps_subunit)
        if jumps_subunit_to_add != set():
            for count, jump_subunit in enumerate(jumps_subunit_to_add, count):
                lines.insert(last_jump + count, 'connect_virtual JUMP%s_to_subunit VRT%s_base SUBUNIT'
                             % (jump_subunit, jump_subunit))
            lines[-1] = \
                lines[-1].strip() + (len(jumps_subunit_to_add) * ' JUMP%s_to_subunit') % tuple(jumps_subunit_to_add)

        if modify_sym_energy_for_cryst:
            # new energy should equal the energy multiplier times the scoring subunit plus additional complex subunits
            # where complex subunits = num_subunits - 1
            # new_energy = 'E = %d*%s + ' % (energy, subunits[0])  # assumes subunits are read in alphanumerical order
            # new_energy += ' + '.join('1*(%s:%s)' % t for t in zip(repeat(subunits[0]), subunits[1:]))
            lines[1] = 'E = 2*%s+%s' \
                % (subunits[0], '+'.join('1*(%s:%s)' % (subunits[0], pair) for pair in subunits[1:]))
        else:
            if not energy:
                energy = len(subunits)
            lines[1] = 'E = %d*%s+%s' \
                % (energy, subunits[0], '+'.join('%d*(%s:%s)' % (energy, subunits[0], pair) for pair in subunits[1:]))

        if not to_file:
            to_file = os.path.join(out_path, '%s.sdf' % self.name)

        with open(to_file, 'w') as f:
            f.write('%s\n' % '\n'.join(lines))
        if count != 0:
            self.log.info('Symmetry Definition File "%s" was missing %d lines, so a fix was attempted. '
                          'Modelling may be affected' % (to_file, count))
        return to_file

    def format_missing_loops_for_design(self, max_loop_length: int = 12, exclude_n_term: bool = True,
                                        ignore_termini: bool = False, **kwargs) \
            -> tuple[list[tuple], dict[int, int], int]:
        """Process missing residue information to prepare for loop modelling files. Assumes residues in pose numbering!

        Args:
            max_loop_length: The max length for loop modelling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term: Whether to exclude the N-termini from modelling due to Remodel Bug
            ignore_termini: Whether to ignore terminal loops in the loop file
        Returns:
            each loop start/end indices, loop and adjacent indices (not all disordered indices) mapped to their
                disordered residue indices, n-terminal residue index
        """
        disordered_residues = self.disorder  # {residue_number: {'from': ,'to': }, ...}
        reference_sequence_length = len(self.reference_sequence)
        # disorder_indices = list(disordered_residues.keys())
        # disorder_indices = []  # holds the indices that should be inserted into the total residues to be modelled
        loop_indices = []  # holds the loop indices
        loop_to_disorder_indices = {}  # holds the indices that should be inserted into the total residues to be modelled
        n_terminal_idx = 0  # initialize as an impossible value
        excluded_disorder = 0  # total residues excluded from loop modelling. Needed for pose numbering translation
        segment_length = 0  # iterate each missing residue
        n_term = False
        loop_start, loop_end = None, None
        for idx, residue_number in enumerate(disordered_residues.keys(), 1):
            segment_length += 1
            if residue_number - 1 not in disordered_residues:  # indicate that this residue_number starts disorder
                # print('Residue number -1 not in loops', residue_number)
                loop_start = residue_number - 1 - excluded_disorder  # - 1 as loop modelling needs existing residue
                if loop_start < 1:
                    n_term = True

            if residue_number + 1 not in disordered_residues:  # the segment has ended
                if residue_number != reference_sequence_length:  # is it not the c-termini?
                    # print('Residue number +1 not in loops', residue_number)
                    # print('Adding loop with length', segment_length)
                    if segment_length <= max_loop_length:  # modelling useful, add to loop_indices
                        if n_term and (ignore_termini or exclude_n_term):  # check if the n_terminus should be included
                            excluded_disorder += segment_length  # sum the exclusion length
                            n_term = False  # we don't have any more n_term considerations
                        else:  # include the segment in the disorder_indices
                            loop_end = residue_number + 1 - excluded_disorder
                            loop_indices.append((loop_start, loop_end))
                            for it, residue_index in enumerate(range(loop_start + 1, loop_end), 1):
                                loop_to_disorder_indices[residue_index] = residue_number - (segment_length - it)
                            # set the start and end indices as out of bounds numbers
                            loop_to_disorder_indices[loop_start], loop_to_disorder_indices[loop_end] = -1, -1
                            if n_term and idx != 1:  # if n-termini and not just start Met
                                n_terminal_idx = loop_end  # save idx of last n-term insertion
                    else:  # modelling not useful, sum the exclusion length
                        excluded_disorder += segment_length
                    # after handling disordered segment, reset increment and loop indices
                    segment_length = 0
                    loop_start, loop_end = None, None
                # residue number is the c-terminal residue
                elif ignore_termini:  # do we ignore termini?
                    if segment_length <= max_loop_length:
                        # loop_end = loop_start + 1 + segment_length  # - excluded_disorder
                        loop_end = residue_number - excluded_disorder
                        loop_indices.append((loop_start, loop_end))
                        for it, residue_index in enumerate(range(loop_start + 1, loop_end), 1):
                            loop_to_disorder_indices[residue_index] = residue_number - (segment_length - it)
                        # don't include start index in the loop_to_disorder map since c-terminal doesn't have attachment
                        loop_to_disorder_indices[loop_end] = -1

        return loop_indices, loop_to_disorder_indices, n_terminal_idx

    # Todo move both of these to Structure/Pose. Requires using .reference_sequence in Structure/ or maybe Pose better
    def make_loop_file(self, out_path: AnyStr = os.getcwd(), **kwargs) -> AnyStr | None:
        """Format a loops file according to Rosetta specifications. Assumes residues in pose numbering!

        The loop file format consists of one line for each specified loop with the format:

        LOOP 779 784 0 0 1

        Where LOOP specifies a loop line, start idx, end idx, cut site (0 lets Rosetta choose), skip rate, and extended

        All indices should refer to existing locations in the structure file so if a loop should be inserted into
        missing density, the density needs to be modelled first before the loop file would work to be modelled. You
        can't therefore specify that a loop should be between 779 and 780 if the loop is 12 residues long since there is
         no specification about how to insert those residues. This type of task requires a blueprint file.

        Args:
            out_path: The location the file should be written
        Keyword Args:
            max_loop_length=12 (int): The max length for loop modelling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term=True (bool): Whether to exclude the N-termini from modelling due to Remodel Bug
            ignore_termini=False (bool): Whether to ignore terminal loops in the loop file
        Returns:
            The path of the file if one was written
        """
        loop_indices, _, _ = self.format_missing_loops_for_design(**kwargs)
        if not loop_indices:
            return
        loop_file = os.path.join(out_path, f'{self.name}.loops')
        with open(loop_file, 'w') as f:
            f.write('%s\n' % '\n'.join(f'LOOP {start} {stop} 0 0 1' for start, stop in loop_indices))

        return loop_file

    def make_blueprint_file(self, out_path: AnyStr = os.getcwd(), **kwargs) -> AnyStr | None:
        """Format a blueprint file according to Rosetta specifications. Assumes residues in pose numbering!

        The blueprint file format is described nicely here:
            https://www.rosettacommons.org/docs/latest/application_documentation/design/rosettaremodel

        In a gist, a blueprint file consists of entries describing the type of design available at each position.

        Ex:
            1 x L PIKAA M   <- Extension

            1 x L PIKAA V   <- Extension

            1 V L PIKAA V   <- Attachment point

            2 D .

            3 K .

            4 I .

            5 L N PIKAA N   <- Attachment point

            0 x I NATAA     <- Insertion

            0 x I NATAA     <- Insertion

            6 N A PIKAA A   <- Attachment point

            7 G .

            0 X L PIKAA Y   <- Extension

            0 X L PIKAA P   <- Extension

        All structural indices must be specified in "pose numbering", i.e. starting with 1 ending with the last residue.
        If you have missing density in the middle, you should not specify those residues that are missing, but keep
        continuous numbering. You can specify an inclusion by specifying the entry index as 0 followed by the blueprint
        directive. For missing density at the n- or c-termini, the file should still start 1, however, the n-termini
        should be extended by prepending extra entries to the structurally defined n-termini entry 1. These blueprint
        entries should also have 1 as the residue index. For c-termini, extra entries should be appended with the
        indices as 0 like in insertions. For all unmodelled entries for which design should be performed, there should
        be flanking attachment points that are also capable of design. Designable entries are seen above with the PIKAA
        directive. Other directives are available. The only location this isn't required is at the c-terminal attachment
        point

        Args:
            out_path: The location the file should be written
        Keyword Args:
            max_loop_length=12 (int): The max length for loop modelling.
                12 is the max for accurate KIC as of benchmarks from T. Kortemme, 2014
            exclude_n_term=True (bool): Whether to exclude the N-termini from modelling due to Remodel Bug
            ignore_termini=False (bool): Whether to ignore terminal loops in the loop file
        Returns:
            The path of the file if one was written
        """
        disordered_residues = self.disorder  # {residue_number: {'from': ,'to': }, ...}
        # trying to remove tags at this stage runs into a serious indexing problem where tags need to be deleted from
        # disordered_residues and then all subsequent indices adjusted.

        # # look for existing tag to remove from sequence and save identity
        # available_tags = find_expression_tags(self.reference_sequence)
        # if available_tags:
        #     loop_sequences = ''.join(mutation['from'] for mutation in disordered_residues)
        #     remove_loop_pairs = []
        #     for tag in available_tags:
        #         tag_location = loop_sequences.find(tag['sequences'])
        #         if tag_location != -1:
        #             remove_loop_pairs.append((tag_location, len(tag['sequences'])))
        #     for tag_start, tag_length in remove_loop_pairs:
        #         for
        #
        #     # untagged_seq = remove_expression_tags(loop_sequences, [tag['sequence'] for tag in available_tags])

        _, disorder_indices, start_idx = self.format_missing_loops_for_design(**kwargs)
        if not disorder_indices:
            return

        residues = self.residues
        # for residue_number in sorted(disorder_indices):  # ensure ascending order, insert dependent on prior inserts
        for residue_index, disordered_residue in disorder_indices.items():
            mutation = disordered_residues.get(disordered_residue)
            if mutation:  # add disordered residue to residues list if they exist
                residues.insert(residue_index - 1, mutation['from'])  # offset to match residues zero-index

        #                 index AA SS Choice AA
        # structure_str   = '%d %s %s'
        # loop_str        = '%d X %s PIKAA %s'
        blueprint_lines = []
        for idx, residue in enumerate(residues, 1):
            if isinstance(residue, Residue):  # use structure_str template
                residue_type = protein_letters_3to1_extended.get(residue.type.title())
                blueprint_lines.append(f'{residue.number} {residue_type} '
                                       f'{f"L PIKAA {residue_type}" if idx in disorder_indices else "."}')
            else:  # residue is the residue type from above insertion, use loop_str template
                blueprint_lines.append(f'{1 if idx < start_idx else 0} X {"L"} PIKAA {residue}')

        blueprint_file = os.path.join(out_path, f'{self.name}.blueprint')
        with open(blueprint_file, 'w') as f:
            f.write('%s\n' % '\n'.join(blueprint_lines))
        return blueprint_file

    def _update_structure_container_attributes(self, **kwargs):
        """Update attributes specified by keyword args for all Structure container members. Entity specific handling"""
        # As this was causing error with keeping mate chains, mates, just return
        return
        # The below code mirrors the _update_structure_container_attributes() from Structure, mius the [1:] slice
        # for structure_type in self.structure_containers:
        #     for structure in getattr(self, structure_type)[1:]:  # only operate on [1:] slice since index 0 is different
        #         for kwarg, value in kwargs.items():
        #             setattr(structure, kwarg, value)

    def _copy_structure_containers(self):
        """Copy all member Structures that reside in Structure containers. Entity specific handling of chains index 0"""
        # self.log.debug('In Entity copy_structure_containers()')
        for structure_type in self.structure_containers:
            structures = getattr(self, structure_type)
            for idx, structure in enumerate(structures[1:], 1):  # only operate on [1:] slice since index 0 is different
                # structures[idx] = copy(structure)
                structure.entity_spawn = True
                new_structure = copy(structure)
                new_structure._captain = self
                structures[idx] = new_structure

    def __copy__(self) -> Entity:  # -> Self Todo python3.11
        # Temporarily remove the _captain attribute for the copy
        # self.log.debug('In Entity copy')
        captain = self._captain
        del self._captain
        other = super().__copy__()
        if self._is_captain:  # If the copier is a captain
            other._captain = None  # Initialize the copy as a captain -> None
        else:
            # If the copy was initiated by the captain, ._captain
            # will be set after this __copy__ return in _copy_structure_containers()
            # This is True if the .entity_spawn attribute is set
            try:  # To delete entity_spawn attribute for the self and the copy (it was copied)
                del self.entity_spawn
                del other.entity_spawn
            except AttributeError:
                # This isn't a captain and a captain didn't initiate the copy
                # We have to make it a captain
                # Add self._captain object to other._captain
                other._captain = captain
                # _make_captain will set _captain as None
                other._make_captain()

        # Set the first chain as the object itself
        other._chains[0] = other
        # Reset the _captain attribute on self as before the copy
        self._captain = captain

        return other

    def __key(self) -> tuple[str, int, ...]:
        return self.name, *self._residue_indices

    def __eq__(self, other: Structure) -> bool:
        if isinstance(other, Entity):
            # The first is True if this is a mate, the second is True if this is a captain
            same_entity = id(self._captain) == id(other) or id(other._captain) == id(self)
        else:
            same_entity = False

        if isinstance(other, Structure):
            return same_entity or self.__key() == other.__key()
        raise NotImplementedError(f'Can\' compare {type(self).__name__} instance to {type(other).__name__} instance')

    # Must define __hash__ in all subclasses that define an __eq__
    def __hash__(self) -> int:
        return hash(self.__key())


def superposition3d(fixed_coords: np.ndarray, moving_coords: np.ndarray, a_weights: np.ndarray = None,
                    quaternion: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Takes two xyz coordinate sets (same length), and attempts to superimpose them using rotations, translations,
    and (optionally) rescale operations to minimize the root mean squared distance (RMSD) between them. The found
    transformation operations should be applied to the "moving_coords" to place them in the setting of the fixed_coords

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    The quaternion_matrix has the last entry storing cos(θ/2) (where θ is the rotation angle). The first 3 entries
    form a vector (of length sin(θ/2)), pointing along the axis of rotation.
    Details: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    MIT License. Copyright (c) 2016, Andrew Jewett
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Args:
        fixed_coords: The coordinates for the 'frozen' object
        moving_coords: The coordinates for the 'mobile' object
        quaternion: Whether to report the rotation angle and axis in Scipy.Rotation quaternion format
    Raises:
        AssertionError: If coordinates are not the same length
    Returns:
        rmsd, rotation/quaternion_matrix, translation_vector
    """
    number_of_points = fixed_coords.shape[0]
    if number_of_points != moving_coords.shape[0]:
        raise ValueError(f'{superposition3d.__name__}: Inputs should have the same size. '
                         f'Input 1={number_of_points}, 2={moving_coords.shape[0]}')

    # convert weights into array
    # if a_weights is None or len(a_weights) == 0:
    # a_weights = np.full((number_of_points, 1), 1.)
    # sum_weights = float(number_of_points)
    # else:  # reshape a_eights so multiplications are done column-wise
    #     a_weights = np.array(a_weights).reshape(number_of_points, 1)
    #     sum_weights = np.sum(a_weights, axis=0)

    # Find the center of mass of each object:
    center_of_mass_fixed = fixed_coords.sum(axis=0)
    center_of_mass_moving = moving_coords.sum(axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    # if sum_weights != 0:
    try:
        center_of_mass_fixed /= number_of_points
        center_of_mass_moving /= number_of_points
    except ZeroDivisionError:
        pass  # the weights are a total of zero which is allowed algorithmically, but not possible

    # Translate the center of mass to the origin
    fixed_coords_at_origin = fixed_coords - center_of_mass_fixed
    moving_coords_at_origin = moving_coords - center_of_mass_moving

    # Calculate the "m" array from the Diamond paper (equation 16)
    m = np.matmul(moving_coords_at_origin.T, fixed_coords_at_origin)

    # Calculate "v" (equation 18)
    # v = np.empty(3)
    # v[0] = m[1][2] - m[2][1]
    # v[1] = m[2][0] - m[0][2]
    # v[2] = m[0][1] - m[1][0]
    v = [m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0]]

    # Calculate "P" (equation 22)
    matrix_p = np.zeros((4, 4))
    # Calculate "q" (equation 17)
    # q = m + m.T - 2*identity_matrix*np.trace(m)
    matrix_p[:3, :3] = m + m.T - 2*identity_matrix*np.trace(m)
    matrix_p[3, :3] = v
    matrix_p[:3, 3] = v
    # [[ q[0][0] q[0][1] q[0][2] v[0] ]
    #  [ q[1][0] q[1][1] q[1][2] v[1] ]
    #  [ q[2][0] q[2][1] q[2][2] v[2] ]
    #  [ v[0]    v[1]    v[2]    0    ]]

    # Calculate "p" - optimal_quat
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below)
    if number_of_points < 2:
        # Specify the default values for p, pPp
        optimal_quat = np.array([0., 0., 0., 1.])  # p = [0,0,0,1]    default value
        pPp = 0.  # = p^T * P * p    (zero by default)
    else:
        # try:
        # The a_eigenvals are returned as 1D array in ascending order; largest is last
        a_eigenvals, aa_eigenvects = np.linalg.eigh(matrix_p)
        # except np.linalg.LinAlgError:
        #     singular = True  # I have never seen this happen
        pPp = a_eigenvals[-1]
        optimal_quat = aa_eigenvects[:, -1]  # pull out the largest magnitude eigenvector
        # normalize the vector
        # (It should be normalized already, but just in case it is not, do it again)
        # optimal_quat /= np.linalg.norm(optimal_quat)

    # Calculate the rotation matrix corresponding to "optimal_quat" which is in scipy quaternion format
    """
    rotation_matrix = np.empty((3, 3))
    rotation_matrix[0][0] = (optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                     -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[1][1] = -(optimal_quat[0]*optimal_quat[0])+(optimal_quat[1]*optimal_quat[1])
                      -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[2][2] = -(optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                      +(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[0][1] = 2*(optimal_quat[0]*optimal_quat[1] - optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][0] = 2*(optimal_quat[0]*optimal_quat[1] + optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][2] = 2*(optimal_quat[1]*optimal_quat[2] - optimal_quat[0]*optimal_quat[3])
    rotation_matrix[2][1] = 2*(optimal_quat[1]*optimal_quat[2] + optimal_quat[0]*optimal_quat[3])
    rotation_matrix[0][2] = 2*(optimal_quat[0]*optimal_quat[2] + optimal_quat[1]*optimal_quat[3])
    rotation_matrix[2][0] = 2*(optimal_quat[0]*optimal_quat[2] - optimal_quat[1]*optimal_quat[3])
    """
    # Alternatively, in modern python versions, this code also works:
    rotation_matrix = Rotation.from_quat(optimal_quat).as_matrix()

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper
    # e0 = np.sum((fixed_coords_at_origin - moving_coords_at_origin) ** 2)
    # sum_sqr_dist = max(0, ((fixed_coords_at_origin-moving_coords_at_origin) ** 2).sum() - 2.*pPp)

    # if sum_weights != 0.:
    try:
        rmsd = np.sqrt(max(0, ((fixed_coords_at_origin-moving_coords_at_origin) ** 2).sum() - 2.*pPp) / number_of_points)
    except ZeroDivisionError:
        rmsd = 0.  # the weights are a total of zero which is allowed algorithmically, but not possible

    # Lastly, calculate the translational offset:
    # Recall that:
    # RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = center_of_mass_fixed[]       and       center_of_mass_moving[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    # a_translate = center_of_mass_fixed - np.matmul(c * rotation_matrix, center_of_mass_moving).T.reshape(3,)

    # Calculate the translation
    translation = center_of_mass_fixed - np.matmul(rotation_matrix, center_of_mass_moving)
    if quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        # rotation_matrix = np.array([p[3], p[0], p[1], p[2]])
        # KM: Disregard above, I am using the scipy version for python continuity which returns X, Y, Z, W
        return rmsd, optimal_quat, translation
    else:
        return rmsd, rotation_matrix, translation


def superposition3d_weighted(fixed_coords: np.ndarray, moving_coords: np.ndarray, a_weights: np.ndarray = None,
                             quaternion: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Takes two xyz coordinate sets (same length), and attempts to superimpose them using rotations, translations,
    and (optionally) rescale operations to minimize the root mean squared distance (RMSD) between them. The found
    transformation operations should be applied to the "moving_coords" to place them in the setting of the fixed_coords

    This function implements a more general variant of the method from:
    R. Diamond, (1988) "A Note on the Rotational Superposition Problem", Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly. The version in the original paper only considers rotation and translation
    and does not allow the coordinates of either object to be rescaled (multiplication by a scalar).
    (Additional documentation can be found at https://pypi.org/project/superpose3d/ )

    The quaternion_matrix has the last entry storing cos(θ/2) (where θ is the rotation angle). The first 3 entries
    form a vector (of length sin(θ/2)), pointing along the axis of rotation.
    Details: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    MIT License. Copyright (c) 2016, Andrew Jewett
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
    OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Args:
        fixed_coords: The coordinates for the 'frozen' object
        moving_coords: The coordinates for the 'mobile' object
        a_weights: Weights for the calculation of RMSD
        quaternion: Whether to report the rotation angle and axis in Scipy.Rotation quaternion format
    Raises:
        AssertionError: If coordinates are not the same length
    Returns:
        rmsd, rotation/quaternion_matrix, translation_vector
    """
    number_of_points = fixed_coords.shape[0]
    if number_of_points != moving_coords.shape[0]:
        raise ValueError(f'{superposition3d.__name__}: Inputs should have the same size. '
                         f'Input 1={number_of_points}, 2={moving_coords.shape[0]}')

    # convert weights into array
    if a_weights is None or len(a_weights) == 0:
        a_weights = np.full((number_of_points, 1), 1.)
        sum_weights = float(number_of_points)
    else:  # reshape a_eights so multiplications are done column-wise
        a_weights = np.array(a_weights).reshape(number_of_points, 1)
        sum_weights = np.sum(a_weights, axis=0)

    # Find the center of mass of each object:
    center_of_mass_fixed = np.sum(fixed_coords * a_weights, axis=0)
    center_of_mass_moving = np.sum(moving_coords * a_weights, axis=0)

    # Subtract the centers-of-mass from the original coordinates for each object
    # if sum_weights != 0:
    try:
        center_of_mass_fixed /= sum_weights
        center_of_mass_moving /= sum_weights
    except ZeroDivisionError:
        pass  # the weights are a total of zero which is allowed algorithmically, but not possible

    aa_xf = fixed_coords - center_of_mass_fixed
    aa_xm = moving_coords - center_of_mass_moving

    # Calculate the "m" array from the Diamond paper (equation 16)
    m = np.matmul(aa_xm.T, (aa_xf * a_weights))

    # Calculate "v" (equation 18)
    v = np.empty(3)
    v[0] = m[1][2] - m[2][1]
    v[1] = m[2][0] - m[0][2]
    v[2] = m[0][1] - m[1][0]

    # Calculate "P" (equation 22)
    matrix_p = np.zeros((4, 4))
    # Calculate "q" (equation 17)
    # q = m + m.T - 2*identity_matrix*np.trace(m)
    matrix_p[:3, :3] = m + m.T - 2*identity_matrix*np.trace(m)
    matrix_p[3, :3] = v
    matrix_p[:3, 3] = v
    # [[ q[0][0] q[0][1] q[0][2] v[0] ]
    #  [ q[1][0] q[1][1] q[1][2] v[1] ]
    #  [ q[2][0] q[2][1] q[2][2] v[2] ]
    #  [ v[0]    v[1]    v[2]    0    ]]

    # Calculate "p" - optimal_quat
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below)
    if number_of_points < 2:
        # Specify the default values for p, pPp
        optimal_quat = np.array([0., 0., 0., 1.])  # p = [0,0,0,1]    default value
        pPp = 0.  # = p^T * P * p    (zero by default)
    else:
        # try:
        a_eigenvals, aa_eigenvects = np.linalg.eigh(matrix_p)
        # except np.linalg.LinAlgError:
        #     singular = True  # I have never seen this happen
        pPp = np.max(a_eigenvals)
        optimal_quat = aa_eigenvects[:, np.argmax(a_eigenvals)]  # pull out the largest magnitude eigenvector
        # normalize the vector
        # (It should be normalized already, but just in case it is not, do it again)
        optimal_quat /= np.linalg.norm(optimal_quat)

    # Calculate the rotation matrix corresponding to "optimal_quat" which is in scipy quaternion format
    """
    rotation_matrix = np.empty((3, 3))
    rotation_matrix[0][0] = (optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                     -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[1][1] = -(optimal_quat[0]*optimal_quat[0])+(optimal_quat[1]*optimal_quat[1])
                      -(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[2][2] = -(optimal_quat[0]*optimal_quat[0])-(optimal_quat[1]*optimal_quat[1])
                      +(optimal_quat[2]*optimal_quat[2])+(optimal_quat[3]*optimal_quat[3])
    rotation_matrix[0][1] = 2*(optimal_quat[0]*optimal_quat[1] - optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][0] = 2*(optimal_quat[0]*optimal_quat[1] + optimal_quat[2]*optimal_quat[3])
    rotation_matrix[1][2] = 2*(optimal_quat[1]*optimal_quat[2] - optimal_quat[0]*optimal_quat[3])
    rotation_matrix[2][1] = 2*(optimal_quat[1]*optimal_quat[2] + optimal_quat[0]*optimal_quat[3])
    rotation_matrix[0][2] = 2*(optimal_quat[0]*optimal_quat[2] + optimal_quat[1]*optimal_quat[3])
    rotation_matrix[2][0] = 2*(optimal_quat[0]*optimal_quat[2] - optimal_quat[1]*optimal_quat[3])
    """
    # Alternatively, in modern python versions, this code also works:
    rotation_matrix = Rotation.from_quat(optimal_quat).as_matrix()

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper
    # e0 = np.sum((aa_xf - aa_xm) ** 2)
    # sum_sqr_dist = max(0, ((aa_xf-aa_xm) ** 2).sum() - 2.*pPp)

    # if sum_weights != 0.:
    try:
        rmsd = np.sqrt(max(0, ((aa_xf-aa_xm) ** 2).sum() - 2.*pPp) / sum_weights)
    except ZeroDivisionError:
        rmsd = 0.  # the weights are a total of zero which is allowed algorithmically, but not possible

    # Lastly, calculate the translational offset:
    # Recall that:
    # RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = center_of_mass_fixed[]       and       center_of_mass_moving[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  a_translate[i]

    # a_translate = center_of_mass_fixed - np.matmul(c * aa_rotate, center_of_mass_moving).T.reshape(3,)

    # return rmsd, aa_rotate, center_of_mass_fixed - np.matmul(aa_rotate, center_of_mass_moving).T.reshape(3,)
    # Calculate the translation
    translation = center_of_mass_fixed - np.matmul(rotation_matrix, center_of_mass_moving)
    if quaternion:  # does the caller want the quaternion?
        # The p array is a quaternion that uses this convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
        # However it seems that the following convention is much more popular:
        # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
        # https://mathworld.wolfram.com/Quaternion.html
        # So I return "q" (a version of "p" using the more popular convention).
        # rotation_matrix = np.array([p[3], p[0], p[1], p[2]])
        # KM: Disregard above, I am using the scipy version for python continuity
        return rmsd, optimal_quat, translation
    else:
        return rmsd, rotation_matrix, translation


def parse_stride(stride_file, **kwargs):
    """From a Stride file, parse information for residue level secondary structure assignment

    Sets:
        self.secondary_structure
    """
    with open(stride_file, 'r') as f:
        stride_output = f.readlines()

    return ''.join(line[24:25] for line in stride_output if line[0:3] == 'ASG')


# @njit
def transform_coordinates(coords: np.ndarray | Iterable, rotation: np.ndarray | Iterable = None,
                          translation: np.ndarray | Iterable | int | float = None,
                          rotation2: np.ndarray | Iterable = None,
                          translation2: np.ndarray | Iterable | int | float = None) -> np.ndarray:
    """Take a set of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the order of
    operations as: rotation, translation, rotation2, translation2

    Args:
        coords: The coordinates to transform, can be shape (number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (3, 3)
        translation: The first translation to apply, expected shape (3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (3, 3)
        translation2: The second translation to apply, expected shape (3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    new_coords = coords.copy()

    if rotation is not None:
        np.matmul(new_coords, np.transpose(rotation), out=new_coords)

    if translation is not None:
        new_coords += translation  # No array allocation, sets in place

    if rotation2 is not None:
        np.matmul(new_coords, np.transpose(rotation2), out=new_coords)

    if translation2 is not None:
        new_coords += translation2

    return coords


# @njit
def transform_coordinate_sets_with_broadcast(coord_sets: np.ndarray,
                                             rotation: np.ndarray = None,
                                             translation: np.ndarray | Iterable | int | float = None,
                                             rotation2: np.ndarray = None,
                                             translation2: np.ndarray | Iterable | int | float = None) \
        -> np.ndarray:
    """Take stacked sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2. Non-efficient memory use

    Args:
        coord_sets: The coordinates to transform, can be shape (number of sets, number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation: The first translation to apply, expected shape (number of sets, 3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation2: The second translation to apply, expected shape (number of sets, 3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_shape = getattr(coord_sets, 'shape', None)
    if set_shape is None or set_shape[0] < 1:
        return coord_sets
    # else:  # Create a new array for the result
    #     new_coord_sets = coord_sets.copy()

    if rotation is not None:
        coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))

    if translation is not None:
        coord_sets += translation  # No array allocation, sets in place

    if rotation2 is not None:
        coord_sets = np.matmul(coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        coord_sets += translation2

    return coord_sets


# @njit
def transform_coordinate_sets(coord_sets: np.ndarray,
                              rotation: np.ndarray = None, translation: np.ndarray | Iterable | int | float = None,
                              rotation2: np.ndarray = None, translation2: np.ndarray | Iterable | int | float = None) \
        -> np.ndarray:
    """Take stacked sets of x,y,z coordinates and transform. Transformation proceeds by matrix multiplication with the
    order of operations as: rotation, translation, rotation2, translation2. If transformation uses broadcasting, for
    efficient memory use, the returned array will be the size of the coord_sets multiplied by rotation. Additional
    broadcasting is not allowed. If that behavior is desired, use "transform_coordinate_sets_with_broadcast()" instead

    Args:
        coord_sets: The coordinates to transform, can be shape (number of sets, number of coordinates, 3)
        rotation: The first rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation: The first translation to apply, expected shape (number of sets, 3)
        rotation2: The second rotation to apply, expected general rotation matrix shape (number of sets, 3, 3)
        translation2: The second translation to apply, expected shape (number of sets, 3)
    Returns:
        The transformed coordinate set with the same shape as the original
    """
    # in general, the np.tensordot module accomplishes this coordinate set multiplication without stacking
    # np.tensordot(a, b, axes=1)  <-- axes=1 performs the correct multiplication with a 3d (3,3,N) by 2d (3,3) matrix
    # np.matmul solves as well due to broadcasting
    set_shape = getattr(coord_sets, 'shape', None)
    if set_shape is None or set_shape[0] < 1:
        return coord_sets

    if rotation is not None:
        new_coord_sets = np.matmul(coord_sets, rotation.swapaxes(-2, -1))
    else:  # Create a new array for the result
        new_coord_sets = coord_sets.copy()

    if translation is not None:
        new_coord_sets += translation  # No array allocation, sets in place

    if rotation2 is not None:
        np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1), out=new_coord_sets)
        # new_coord_sets[:] = np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1))
        # new_coord_sets = np.matmul(new_coord_sets, rotation2.swapaxes(-2, -1))

    if translation2 is not None:
        new_coord_sets += translation2

    return new_coord_sets
