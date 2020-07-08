import os
import sys
import copy
import json
import argparse
import pickle
import pandas as pd
import numpy as np
from itertools import repeat
import PDB
import SymDesignUtils as SDUtils
import PathUtils as PUtils
import AnalyzeMutatedSequences as Ams
# import CmdUtils as CUtils
logger = SDUtils.start_log(__name__)

# Globals
metric_master = {'buns_heavy_total': 'Buried unsaturated hydrogen bonds between heavy atoms in the pose',
                 'buns_hpol_total': 'Buried unsaturated hydrogen bonds between hydrogen atoms in the pose',
                 'buns_total': 'Buried unsaturated hydrogen bonds in the pose',
                 'contact_count': 'Number of carbon carbon contacts across interface residues',
                 'cst_weight': 'Total weight of constraints used to keep the pose from moving in 3D space',
                 'fsp_energy': 'Total weight of sequence constraints used to favor certain amino acids in design. '
                               'Only some protocols have values',
                 'fsp_total_stability': 'fsp_energy + total pose energy',
                 'full_stability': 'total pose energy (essentially REU from above)',
                 'int_area_hydrophobic': 'solvent accessible surface total interface area hydrophobic',
                 'int_area_polar': 'solvent accessible surface total interface area polar',
                 'int_area_res_summary_hydrophobic_A_oligomer': 'Sum of each interface residues individual area for '
                                                                'oligomer A - hydrophobic',
                 'int_area_res_summary_hydrophobic_B_oligomer': 'Sum of each interface residues individual area for '
                                                                'oligomer B - hydrophobic',
                 'int_area_res_summary_polar_A_oligomer': 'Sum of each interface residues individual area for '
                                                          'oligomer A - polar',
                 'int_area_res_summary_polar_B_oligomer': 'Sum of each interface residues individual area for '
                                                          'oligomer B - polar',
                 'int_area_res_summary_total_A_oligomer': 'Sum of each interface residues individual area for '
                                                          'oligomer A - total',
                 'int_area_res_summary_total_B_oligomer': 'Sum of each interface residues individual area for '
                                                          'oligomer B - total',
                 'int_area_total': 'solvent accessible surface total interface area total',
                 'int_connectivity_A': '"Interface connection chainA to the rest of the protein',
                 'int_connectivity_B': '"Interface connection chainB to the rest of the protein',
                 'int_energy_context_A_oligomer': 'interface energy of the A oligomer',
                 'int_energy_context_B_oligomer': 'interface energy of the B oligomer',
                 'int_energy_context_complex': 'interface energy of the complex',
                 'int_energy_res_summary_A_oligomer': 'Sum of each interface residues individual energy for the '
                                                      'oligomer A',
                 'int_energy_res_summary_B_oligomer': 'Sum of each interface residues individual energy for the '
                                                      'oligomer B',
                 'int_energy_res_summary_complex': 'Sum of each interface residues individual energy for the complex',
                 'int_energy_res_summary_delta': 'delta of res summary energy complex and oligomer',
                 'int_energy_res_summary_oligomer': 'sum of oligomer res summary energy',
                 'shape_complementarity': 'interface shape complementarity. Measure of fit between two surfaces',
                 'int_separation': 'distance between the mean atoms on two sides of a interface',
                 'number_hbonds': 'The number of h-bonding residues present in the interface',
                 'ref': 'Rosetta Energy Term - A metric for the unfolded protein energy and some sequence fitting '
                        'corrections to the score function',
                 'rmsd': 'Root Mean Square Deviation of all CA atoms between a reference and design state',
                 'protocol': '"Protocol I have created to search sequence space',
                 'nanohedra_score': 'The Nanohedra score that Josh outputs from Nanohedra docking',
                 'fragment_z_score_total': 'The sum of all fragments Z-Scores',
                 'unique_fragments': 'The number of unique fragments placed on the pose',
                 'total_interface_residues': 'The number of interface residues found in the pose',
                 'interface_b_factor_per_res': 'The average B factor for each atom in every interface residue in the '
                                               'pose',
                 'REU': 'Rosetta Energy Units. Always 0. We can disregard',
                 'buns_asu': 'Buried unsaturated hydrogen bonds. This column helps with buns_total '
                             '- can be disregarded',
                 'buns_asu_hpol': 'Buried unsaturated hydrogen bonds. This column helps with buns_total '
                                  '- can be disregarded',
                 'buns_nano': 'Buried unsaturated hydrogen bonds. This column helps with buns_total '
                              '- can be disregarded',
                 'buns_nano_hpol': 'Buried unsaturated hydrogen bonds. This column helps with buns_total '
                                   '- can be disregarded',
                 'int_area_asu_hydrophobic': 'solvent accessible surface asu interface area hydrophobic '
                                             '- can likely remove',
                 'int_area_asu_polar': 'solvent accessible surface asu interface area polar '
                                       '- can likely remove',
                 'int_area_asu_total': 'solvent accessible surface asu interface area total '
                                       '- can likely remove',
                 'int_area_ex_asu_hydrophobic': 'solvent accessible surface extra-asu interface area hydrophobic '
                                                '- can likely remove',
                 'int_area_ex_asu_polar': 'solvent accessible surface extra-asu interface area polar '
                                          '- can likely remove',
                 'int_area_ex_asu_total': 'solvent accessible surface extra-asu interface area total '
                                          '- can likely remove',
                 'int_connectivity1': 'Old connectivity - DEPRECIATED',
                 'int_connectivity2': 'Old connectivity - DEPRECIATED',
                 'int_energy_context_asu': 'interface energy of the ASU',
                 'int_energy_context_unbound': 'interface energy of the unbound',
                 'coordinate_constraint': 'Same as cst_weight',
                 'int_energy_res_summary_asu': 'Sum of each interface residues individual energy for the ASU '
                                               '- DEPRECIATED',
                 'int_energy_res_summary_unbound': 'Sum of each interface residues individual energy for the unbound '
                                                   '- DEPRECIATED',
                 'interaction_energy': 'interaction energy between two sets of residues '
                                       '(excludes intra-residue terms I believe) I think this is left over '
                                       '- DEPRECIATED',
                 'interaction_energy_asu': 'interaction energy between two sets of residues in ASU state '
                                           '(excludes intra-residue terms I believe) - DEPRECIATED',
                 'interaction_energy_oligomerA': 'interaction energy between two sets of residues in oligomerA '
                                                 '(excludes intra-residue terms I believe)',
                 'interaction_energy_oligomerB': 'interaction energy between two sets of residues in oligomerB '
                                                 '(excludes intra-residue terms I believe)',
                 'interaction_energy_unbound': 'interaction energy between two sets of residues in unbound state '
                                               '(excludes intra-residue terms I believe) - DEPRECIATED',
                 'res_type_constraint': 'Same as fsp_energy',
                 'time': 'Time for the protocol to complete',
                 'hbonds_res_selection_complex': 'The specific hbonds present in the Nanohedra pose',
                 'hbonds_res_selection_oligomer_A': 'The specific hbonds present in the oligomeric pose A',
                 'hbonds_res_selection_oligomer_B': 'The specific hbonds present in the oligomeric pose B',
                 'dslf_fa13': 'Rosetta Energy Term - disulfide bonding',
                 'fa_atr': 'Rosetta Energy Term - lennard jones full atom atractive forces',
                 'fa_dun': 'Rosetta Energy Term - dunbrack rotamer library statistical probability',
                 'fa_elec': 'Rosetta Energy Term - full atom electrostatic forces',
                 'fa_intra_rep': 'Rosetta Energy Term - lennard jones full atom intra-residue repulsive forces',
                 'fa_intra_sol_xover4': 'Rosetta Energy Term - full atom intra-residue solvent forces',
                 'fa_rep': 'Rosetta Energy Term - lennard jones full atom repulsive forces',
                 'fa_sol': 'Rosetta Energy Term - full atom solvent forces',
                 'hbond_bb_sc': 'Rosetta Energy Term - backbone/sidechain hydrogen bonding',
                 'hbond_lr_bb': 'Rosetta Energy Term - long range backbone hydrogen bonding',
                 'hbond_sc': 'Rosetta Energy Term - sidechain hydrogen bonding',
                 'hbond_sr_bb': 'Rosetta Energy Term - short range backbone hydrogen bonding',
                 'lk_ball_wtd': 'Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
                 'omega': 'Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
                 'p_aa_pp': '"Rosetta Energy Term - statistical probability of an amino acid given angles phi',
                 'pro_close': 'Rosetta Energy Term - to favor closing of proline rings',
                 'rama_prepro': 'Rosetta Energy Term - amino acid dependent term to favor certain ramachandran angles '
                                'on residue before prolines',
                 'yhh_planarity': 'Rosetta Energy Term - to favor planarity of tyrosine hydrogen'}
necessary_metrics = {'buns_asu_hpol', 'buns_nano_hpol', 'buns_asu', 'buns_nano', 'buns_total', 'contact_count',
                     'cst_weight', 'fsp_energy', 'int_area_hydrophobic', 'int_area_polar',
                     'int_area_total', 'int_connectivity_A', 'int_connectivity_B', 'int_energy_context_A_oligomer',
                     'int_energy_context_B_oligomer', 'int_energy_context_complex',
                     'int_energy_res_summary_A_oligomer',
                     'int_energy_res_summary_B_oligomer', 'int_energy_res_summary_complex',
                     'int_separation',
                     'interaction_energy_complex', 'ref', 'rmsd', 'shape_complementarity',
                     'hbonds_res_selection_complex', 'hbonds_res_selection_oligomer_A',
                     'hbonds_res_selection_oligomer_B'}
# 'fsp_total_stability', 'full_stability',
# 'int_area_res_summary_hydrophobic_A_oligomer', 'int_area_res_summary_hydrophobic_B_oligomer',
# 'int_area_res_summary_polar_A_oligomer', 'int_area_res_summary_polar_B_oligomer',
# 'int_area_res_summary_total_A_oligomer', 'int_area_res_summary_total_B_oligomer',
# 'int_energy_res_summary_delta', 'number_hbonds', 'total_interface_residues', 'int_energy_context_delta',
# 'average_fragment_z_score', 'nanohedra_score', 'unique_fragments', 'interface_b_factor_per_res',
# 'int_energy_res_summary_oligomer', 'int_energy_context_oligomer',

final_metrics = {'buns_heavy_total', 'buns_hpol_total', 'buns_total', 'contact_count', 'cst_weight', 'fsp_energy',
                 'int_area_hydrophobic', 'int_area_polar',
                 'int_area_total', 'int_connectivity_A', 'int_connectivity_B', 'int_energy_context_A_oligomer',
                 'int_energy_context_B_oligomer', 'int_energy_context_complex', 'int_energy_context_delta',
                 'int_energy_context_oligomer', 'int_energy_res_summary_A_oligomer',
                 'int_energy_res_summary_B_oligomer', 'int_energy_res_summary_complex',
                 'int_energy_res_summary_delta', 'int_energy_res_summary_oligomer', 'int_separation',
                 'interaction_energy_complex', 'number_hbonds', 'ref', 'rmsd', 'shape_complementarity',
                 'nanohedra_score', 'average_fragment_z_score', 'unique_fragments', 'total_interface_residues',
                 'interface_b_factor_per_res'}
# 'fsp_total_stability', 'full_stability',
# 'int_area_res_summary_hydrophobic_A_oligomer', 'int_area_res_summary_hydrophobic_B_oligomer',
# 'int_area_res_summary_polar_A_oligomer', 'int_area_res_summary_polar_B_oligomer',
# 'int_area_res_summary_total_A_oligomer', 'int_area_res_summary_total_B_oligomer',
rename_columns = {'int_sc': 'shape_complementarity', 'int_sc_int_area': 'int_area',  # 'total_score': 'REU',
                  'int_sc_median_dist': 'int_separation', 'relax_switch': 'protocol',
                  'full_stability': 'full_stability_complex', 'no_constraint_switch': 'protocol',  # 'decoy': 'design',
                  'limit_to_profile_switch': 'protocol', 'combo_profile_switch': 'protocol',
                  'favor_profile_switch': 'protocol', 'consensus_design_switch': 'protocol'}
remove_columns = ['decoy', 'symmetry_switch', 'oligomer_switch', 'total_score']
save_columns = ['protocol']
# sum columns
summation_pairs = {'buns_hpol_total': ('buns_asu_hpol', 'buns_nano_hpol'),
                   'buns_heavy_total': ('buns_asu', 'buns_nano'),
                   'int_energy_context_oligomer':
                       ('int_energy_context_A_oligomer', 'int_energy_context_B_oligomer'),
                   'int_energy_res_summary_oligomer':
                       ('int_energy_res_summary_A_oligomer', 'int_energy_res_summary_B_oligomer'),
                   'full_stability_oligomer': ('full_stability_oligomer_A', 'full_stability_oligomer_B')}  # ,
#                    'hbonds_oligomer': ('hbonds_res_selection_oligomer_A', 'hbonds_res_selection_oligomer_A')}
# subtract columns using tuple [0] - [1] to make delta column
delta_pairs = {'int_energy_context_delta': ('int_energy_context_complex', 'int_energy_context_oligomer'),
               'int_energy_res_summary_delta': (
               'int_energy_res_summary_complex', 'int_energy_res_summary_oligomer'),
               'full_stability_delta': ('full_stability', 'full_stability_oligomer')}
                # TODO P432 full_stability'_complex'

#                'number_hbonds': ('hbonds_res_selection_complex', 'hbonds_oligomer')}
drop_unneccessary = ['int_area_asu_hydrophobic', 'int_area_asu_polar', 'int_area_asu_total',
                     'int_area_ex_asu_hydrophobic', 'int_area_ex_asu_polar', 'int_area_ex_asu_total',
                     'buns_asu', 'buns_asu_hpol', 'buns_nano', 'buns_nano_hpol', 'int_connectivity1',
                     'int_connectivity2', 'int_energy_context_asu', 'int_energy_context_unbound',
                     'coordinate_constraint', 'int_energy_res_summary_asu', 'int_energy_res_summary_unbound',
                     'interaction_energy', 'interaction_energy_asu', 'interaction_energy_oligomerA',
                     'interaction_energy_oligomerB', 'interaction_energy_unbound', 'res_type_constraint', 'time', 'REU'
                     'full_stability_complex', 'full_stability_oligomer'
                     'int_area_res_summary_hydrophobic_A_oligomer', 'int_area_res_summary_hydrophobic_B_oligomer',
                     'int_area_res_summary_polar_A_oligomer', 'int_area_res_summary_polar_B_oligomer',
                     'int_area_res_summary_total_A_oligomer', 'int_area_res_summary_total_B_oligomer']
rosetta_terms = ['lk_ball_wtd', 'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'yhh_planarity', 'dslf_fa13',
                 'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb']  # 'ref'
specific_columns_to_add = ['int_energy_res_summary_delta', 'int_energy_context_delta',
                           'shape_complementarity',
                           'buns_total', 'contact_count', 'interaction_energy_complex', 'int_area_hydrophobic',
                           'int_area_polar',
                           'int_area_total', 'shape_complementarity', 'number_hbonds']
# 'full_stability',

def clean_index(row):  # UNUSED
    row.index = row.index.split('_')[-1]
    return row


def scores(file):  # UNUSED DEPRECIATED
    with open(file, 'r') as f:
        all_lines = f.readlines()

    all_scores = []
    for line in all_lines:
        if line[:6] == 'SCORE:':
            all_scores.append(line.lstrip('SCORE:').rstrip().split())

    df = pd.DataFrame(all_scores[1:], columns=all_scores[0])
    for column in df.columns:
        if column == 'description':
            for entry in range(len(df[column])):
                df.loc[entry, column] = df.loc[entry, column].split('_')[-1]
            #             df[column] = df[column].rename('design')
            df = df.set_index(['description'])
        else:
            df[column] = df[column].astype(float)
    df = df.rename(columns={'total_score': 'REU', 'int_sc': 'shape_complementarity', 'int_sc_int_area': 'int_area',
                            'int_sc_median_dist': 'int_separation', 'description': 'design'})
    # 'acc': 'atomic_contacts', 'average_degree': 'int_connectivity', 'interface_rotamer_quality':
    # 'interface_rotamer_quality', 'rmsd': 'rmsd', 'sasa_hydrophobic_filter': 'hydrophobic_sasa', 'sasa_polar_filter':
    # 'polar_sasa', 'stability_pure': , 'stability_score_full': , 'stability_without_pssm': '', 'timer': 'time',

    return df


def design_mutations_for_metrics(design_directory, wild_type_file=None):  # DEPRECIATED
    """Given a design directory, find the Wild-type sequence and mutations made in comparison

    Args:
        design_directory (DesignDirectory): Single DesignDirectory object
    Keyword Args:
        wildtype=None (str): The location of a wildtype PDB file
    Returns:
        parsed_design_mutations (dict): {'file_name': {chain_id: {mutation_index: ('From AA', 'To AA'), ...}, ...}, ...}
    """
    if not wild_type_file:
        wild_type_file = Ams.get_wildtype_file(design_directory)
    all_design_files = SDUtils.get_directory_pdb_file_paths(design_directory.design_pdbs)

    return Ams.generate_mutations(all_design_files, wild_type_file, pose_num=True)


def design_mutations_for_sequence(design_directory, wild_type_file=None):  # DEPRECIATED
    """Given a design directory, find the Wild-type sequence and mutations made in comparison

    Args:
        design_directory (DesignDirectory): Single DesignDirectory object
    Keyword Args:
        wildtype=None (str): The location of a wildtype PDB file
    Returns:
        parsed_design_mutations (dict): {'file_name': {chain_id: {mutation_index: ('From AA', 'To AA'), ...}, ...}, ...}
    """
    if not wild_type_file:
        wild_type_file = Ams.get_wildtype_file(design_directory)
    all_design_files = SDUtils.get_directory_pdb_file_paths(design_directory.design_pdbs)

    return Ams.generate_mutations(all_design_files, wild_type_file)


def read_scores(file):
    """Take a json formatted score.sc file and incorporate into dictionary object

    Args:
        file (str): Location on disk of scorefile
    Returns:
        score_dict (dict): {design_name: {all_score_metric_keys: all_acore_metric_values}, ...}
    """
    with open(file, 'r') as f:
        score_dict = {}
        for score in f.readlines():
            entry = json.loads(score)
            if entry['decoy'].split('_')[-1] not in score_dict:
                score_dict[entry['decoy'].split('_')[-1]] = entry
            else:
                score_dict[entry['decoy'].split('_')[-1]].update(entry)

    return score_dict


def join_columns(x):
    """Combine columns in a dataframe with the same column name"""

    return ','.join(x[x.notnull()].astype(str))


def sum_columns_to_new_column(df, column_dict):
    for column in column_dict:
        try:
            df[column] = df[column_dict[column][0]] + df[column_dict[column][1]]
        except KeyError:
            pass

    return df


def subtract_columns_to_new_column(df, column_dict):
    for column in column_dict:
        try:
            df[column] = df[column_dict[column][1]] - df[column_dict[column][0]]
        except KeyError:
            pass

    return df


def drop_extras(df, drop_unneccessary):
    for unnecc in drop_unneccessary:
        try:
            df = df.drop(unnecc, axis=1)
        except KeyError:
            pass
    return df


def remove_pdb_prefixes(pdb_dict):
    clean_key_dict = {}
    for key in pdb_dict:
        new_key = key.split('_')[-1]
        clean_key_dict[new_key] = pdb_dict[key]

    return clean_key_dict


# from table 2 of Miller et al. 1987, sidechain is broken down into polar and non-polar
gxg_sasa = {'A': {'backbone': 46, 'polar': 0, 'non-polar': 67}, 'R': {'backbone': 45, 'polar': 107, 'non-polar': 89},
            'N': {'backbone': 45, 'polar': 69, 'non-polar': 44}, 'D': {'backbone': 45, 'polar': 58, 'non-polar': 48},
            'C': {'backbone': 36, 'polar': 69, 'non-polar': 35}, 'Q': {'backbone': 45, 'polar': 91, 'non-polar': 53},
            'E': {'backbone': 45, 'polar': 77, 'non-polar': 61}, 'G': {'backbone': 85, 'polar': 0, 'non-polar': 0},
            'H': {'backbone': 43, 'polar': 49, 'non-polar': 102}, 'I': {'backbone': 42, 'polar': 0, 'non-polar': 142},
            'L': {'backbone': 43, 'polar': 0, 'non-polar': 137}, 'K': {'backbone': 44, 'polar': 48, 'non-polar': 119},
            'M': {'backbone': 44, 'polar': 43, 'non-polar': 117}, 'F': {'backbone': 43, 'polar': 0, 'non-polar': 175},
            'P': {'backbone': 38, 'polar': 0, 'non-polar': 105}, 'S': {'backbone': 42, 'polar': 36, 'non-polar': 44},
            'T': {'backbone': 44, 'polar': 23, 'non-polar': 74}, 'Y': {'backbone': 42, 'polar': 43, 'non-polar': 144},
            'V': {'backbone': 43, 'polar': 0, 'non-polar': 117}}


def total_gxg_sasa(aa):
    s = 0
    for value in gxg_sasa[aa]:
        s += gxg_sasa[aa][value]

    return s


def sidechain_gxg_sasa(aa):
    s = 0
    for value in gxg_sasa[aa]:
        if value != 'backbone':
            s += gxg_sasa[aa][value]

    return s


def calc_relative_sasa(aa, sasa, sidechain=False):
    if sidechain:
        ref = sidechain_gxg_sasa(aa)
    else:
        ref = total_gxg_sasa(aa)

    return round(sasa / ref, 2)


def hot_spot(residue_dict, energy=-1.5):
    for res in residue_dict:
        if residue_dict[res]['energy'] <= energy:
            residue_dict[res]['hot_spot'] = 1
        else:
            residue_dict[res]['hot_spot'] = 0

    return residue_dict


def hbond_processing(score_dict, columns):
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        columns (list): [hbonds_res_selection_complex, hbonds_res_selection_oligomer_A, hbonds_res_selection_oligomer_A]
    Returns:
        hbond_dict (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    """
    hbond_dict = {}
    for entry in score_dict:
        entry_dict = {}
        for column in columns:
            hbonds = score_dict[entry][column].split(',')
            for i in range(len(hbonds)):
                if hbonds[i] != '':
                    hbonds[i] = int(hbonds[i][:-1])  # remove chain ID off last index
            entry_dict[column.split('_')[-1]] = set(hbonds)
        if entry_dict:
            hbond_dict[entry] = list((entry_dict['complex'] - entry_dict['A']) - entry_dict['B'])

    return hbond_dict


def residue_processing(score_dict, mutations, chain_offset, hbonds=None):
    """Process Residue Metrics from Rosetta score dictionary

    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        mutations (dict): {'0001': {mutation_index: {'from': 'A', 'to: 'K'}, ...}, ...}
        chain_offset (dict): {'A': 0, 'B': 102}
    Keyword Args:
        hbonds=None (list): [34, 54, 67, 68, 106, 178]
    Returns:
        residue_dict (dict): {'0001': {15: {'aa': 'T', 'energy': -2.771, 'bsa_polar': 13.987, 'bsa_hydrophobic': 22.29,
         'bsa_total': 36.278, 'hbond': 0, 'core': 0, 'rim': 1, 'support': 0, 'hot_spot': 1}, ...}, ...}
    """
    dict_template = {'aa': None, 'energy': {'complex': 0, 'A': 0, 'B': 0, 'fsp': 0, 'cst': 0},
                     'sasa': {'polar': {'complex': 0, 'A': 0, 'B': 0}, 'hydrophobic': {'complex': 0, 'A': 0, 'B': 0},
                              'total': {'complex': 0, 'A': 0, 'B': 0}}, 'hbond': 0, 'core': 0, 'rim': 0, 'support': 0}
    #                  , 'hot_spot': 0}
    total_residue_dict = {}
    for entry in score_dict:
        residue_dict = {}
        for key, value in score_dict[entry].items():
            if key.startswith('per_res_'):
                metadata = key.split('_')
                res = int(metadata[-1])
                r_type = metadata[2]  # energy or sasa
                pose_state = metadata[-2]  # oligomer or complex
                if pose_state == 'oligomer':
                    pose_state = metadata[-3]  # get oligomer chain instead
                    # if pose_state != 'A':
                    res += chain_offset[pose_state]
                if res not in residue_dict:
                    residue_dict[res] = {'aa': None, 'energy': {'complex': 0, 'oligomer': 0, 'fsp': 0, 'cst': 0},
                                         'sasa': {'polar': {'complex': 0, 'oligomer': 0},
                                                  'hydrophobic': {'complex': 0, 'oligomer': 0},
                                                  'total': {'complex': 0, 'oligomer': 0}}, 'hbond': 0, 'core': 0,
                                         'rim': 0, 'support': 0}  # , 'hot_spot': 0}
                #                     residue_dict[res] = dict_template.copy()
                if r_type == 'sasa':
                    # Ex. per_res_sasa_hydrophobic_A_oligomer_15 or per_res_sasa_hydrophobic_complex_15
                    polarity = metadata[3]
                    residue_dict[res][r_type][polarity][pose_state] = round(value, 3)
                else:
                    # Ex. per_res_energy_A_oligomer_15 or per_res_energy_complex_15
                    residue_dict[res][r_type][pose_state] = round(value, 3)
        #             elif key.startswith('hbonds_res_selection'):
        #                 hbonds = score_dict[entry][key].split(',')
        #                 for i in range(len(hbonds)):
        #                     # remove chain ID off last index
        #                     hbonds[i] = int(hbonds[i][:-1])
        #                 hbond_dict[key.split('_')[-1]] = hbonds
        #         if hbond_dict:
        #             num_hbonds = len(hbond_dict['complex']) - len(hbond_dict['A']) - len(hbond_dict['B'])
        if residue_dict:
            for res in residue_dict:
                try:
                    residue_dict[res]['aa'] = mutations[entry][res]
                except KeyError:
                    # fill the value with the wild_type sequence
                    residue_dict[res]['aa'] = mutations['ref'][res]
                if hbonds:
                    if res in hbonds:
                        residue_dict[res]['hbond'] = 1
                residue_dict[res]['energy'] = residue_dict[res]['energy']['complex'] - \
                                                  residue_dict[res]['energy']['A'] - residue_dict[res]['energy']['B']
                # residue_dict[res]['energy'] = residue_dict[res]['energy']['complex'] - residue_dict[res]['energy']['oligomer']  # - residue_dict[res]['energy']['fsp']
                #     if residue_dict[res]['energy'] <= hot_spot_energy:
                #         residue_dict[res]['hot_spot'] = 1
                # oligomer_sasa = residue_dict[res]['sasa']['total']['oligomer']
                rel_oligomer_sasa = calc_relative_sasa(residue_dict[res]['aa'],
                                                       residue_dict[res]['sasa']['total']['oligomer'])
                #     polarities = [polarity for polarity in residue_dict[res]['sasa']]
                #     for polarity in residue_dict[res]['sasa']:
                #         residue_dict[res]['sasa_' + polarity] = residue_dict[res]['sasa'][polarity]['complex'] \
                #                                                 - residue_dict[res]['sasa'][polarity]['oligomer']
                #         polarities.append(polarity)
                for polarity in residue_dict[res]['sasa']:
                    residue_dict[res]['bsa_' + polarity] = round(
                        residue_dict[res]['sasa'][polarity]['oligomer'] - residue_dict[res]['sasa'][polarity][
                            'complex'], 2)
                    # if residue_dict[res]['bsa_' + polarity] < 0:
                    #     residue_dict[res]['bsa_' + polarity] = 0
                rel_sasa = calc_relative_sasa(residue_dict[res]['aa'], residue_dict[res]['bsa_total'])
                if rel_oligomer_sasa < 0.25:
                    residue_dict[res]['support'] = 1
                elif rel_sasa > 0.25:
                    residue_dict[res]['rim'] = 1
                else:
                    residue_dict[res]['core'] = 1
                residue_dict[res].pop('sasa')
            total_residue_dict[entry] = residue_dict

    return total_residue_dict


def per_res_metric(divergence_dict, key='jsd'):
    """Find Metric Value/Residue specified by key

    Args:
        divergence_dict (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    Keyword Args:
        key='jsd' (str): Name of the residue metric to average
    Returns:
        jsd_per_res (float): 0.367
    """
    s = 0.0
    for residue in divergence_dict:
        s += divergence_dict[residue][key]

    return round(s / len(divergence_dict), 3)


def gather_fragment_metrics(_des_dir):
    """Gather docking metrics from Nanohedra output"""
    with open(os.path.join(_des_dir.path, PUtils.frag_file), 'r') as f:
        frag_match_info_file = f.readlines()
        residue_cluster_dict = {}
        for line in frag_match_info_file:
            if line[:12] == 'Cluster ID: ':
                cluster = line[12:].split()[0].strip().replace('i', '').replace('j', '').replace('k', '')
            if line[:17] == 'Overlap Z-Value: ':
                residue_cluster_dict[cluster] = float(line[17:].strip())
            if line[:17] == 'Nanohedra Score: ':
                nanohedra_score = float(line[17:].strip())
        #             if line[:39] == 'Unique Interface Fragment Match Count: ':
        #                 int_match = int(line[39:].strip())
        #             if line[:39] == 'Unique Interface Fragment Total Count: ':
        #                 int_total = int(line[39:].strip())
        fragment_z_total = 0
        for cluster in residue_cluster_dict:
            fragment_z_total += residue_cluster_dict[cluster]
        num_fragments = len(residue_cluster_dict)
        ave_z = fragment_z_total / num_fragments

    return {'nanohedra_score': nanohedra_score, 'average_fragment_z_score': ave_z,
            'unique_fragments': num_fragments}  # , 'int_total': int_total}


def datafame_permutation_test(grouped_df, diff_df, group1_samples=None, compare='mean', permutations=1000):
    """From a two group dataframe, compare  run a permutation test and compare """
    # permut_array = [grouped_df.iloc[:group1_samples, :].ave().sub(
    #     grouped_df.iloc[:group1_samples, :].ave()) for i in range(permutations)]
    permut_array = []
    for i in range(permutations):
        grouped_df.sample(len(grouped_df))
        # permut_array.append(grouped_df.iloc[:group1_samples, :].ave().sub(grouped_df.iloc[:group1_samples, :].ave()))
        permut_array.append(grouped_df.iloc[:group1_samples, :].getattr(compare)().sub(
            grouped_df.iloc[:group1_samples, :].getattr(compare)()))
    difference_pvalue = {}
    for column in list(diff_df.columns):
        greater_count = 0
        # check how many permutations have a greater value than the measured difference
        for permut in permut_array:
            if permut[column].value > diff_df[column]:
                greater_count += 1
        difference_pvalue[column] = min(greater_count/permutations, 1 - greater_count/permutations)


def analyze_output(des_dir, delta_refine=False, merge_residue_data=False, debug=False, print_output=False):
    """Retrieve all score information from a design directory and write results to .csv file

    Args:
        des_dir (DesignDirectory): DesignDirectory object
    Keyword Args:
        delta_refine (bbol): Whether to compute DeltaG for residues
        merge_residue_data (bool): Whether to incorporate residue data into Pose dataframe
        debug=False (bool): Whether to debug output
        print_output=False (bool): Whether to log the output to stdout
    Returns:
        scores_df (Dataframe): Dataframe containing the average values from the input design directory
    """
    # Log output
    if debug:
        global logger
    else:
        logger = SDUtils.start_log(name=__name__, handler=2, level=2,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    logger.info('Processing directory \'%s\'' % des_dir.path)
    # Set up pose, ensure proper input
    global remove_columns, rename_columns
    _remove_columns = copy.deepcopy(remove_columns)
    _rename_columns = copy.deepcopy(rename_columns)
    all_design_scores = read_scores(os.path.join(des_dir.scores, PUtils.scores_file))
    scores_df = pd.DataFrame(all_design_scores).T

    # Gather all columns into specific groups for processing and formatting
    report_columns = {}
    per_res_columns = []
    hbonds_columns = []
    for column in list(scores_df.columns):
        if column.startswith('R_'):
            report_columns[column] = column.replace('R_', '')
        elif column.startswith('per_res_'):
            per_res_columns.append(column)
        elif column.startswith('hbonds_res_selection'):
            hbonds_columns.append(column)
    _rename_columns.update(report_columns)
    _rename_columns.update({'R_int_sc': 'shape_complementarity'})  # TODO remove when metrics protocol is changed
    res_columns = hbonds_columns + per_res_columns
    _remove_columns += res_columns + save_columns

    # Format columns
    scores_df = scores_df.rename(columns=_rename_columns)
    scores_df = scores_df.groupby(level=0, axis=1).apply(lambda x: x.apply(join_columns, axis=1))
    # Check proper input
    # for entry in scores_df.columns:  # all_design_scores:
    metric_set = necessary_metrics.copy() - set(scores_df.columns)  # all_design_scores[entry].keys())
    assert metric_set == set(), logger.critical('Pose \'%s\' missing metrics: %s which make this pose unworkable!'
                                                % (str(des_dir), metric_set))
    # CLEAN: Create new columns, remove unneeded columns, create protocol dataframe
    strings_df = scores_df[save_columns]
    indices = strings_df.index.to_list()
    logger.debug('Indices: %s' % str(indices))
    for i, stage in enumerate(PUtils.stage):
        if stage in indices:
            strings_df.at[PUtils.stage[i], 'protocol'] = PUtils.stage[i]  # TODO remove upon future script deployment
            # strings_df.at[PUtils.stage[5], 'protocol'] = PUtils.stage[5]  # TODO remove upon future script deployment
    logger.debug(strings_df)
    scores_df = scores_df.replace('', np.nan)
    scores_df = scores_df.drop(_remove_columns, axis=1).astype(float)
    if delta_refine:
        scores_df = scores_df.sub(scores_df.loc[PUtils.stage[1], ])
    scores_df = sum_columns_to_new_column(scores_df, summation_pairs)
    scores_df = subtract_columns_to_new_column(scores_df, delta_pairs)
    scores_df = drop_extras(scores_df, drop_unneccessary)
    scores_df = drop_extras(scores_df, rosetta_terms)

    # Get design specific wild_type and design files
    wild_type_file = Ams.get_wildtype_file(des_dir)
    wt_sequence = Ams.get_pdb_sequences(wild_type_file)
    all_design_files = SDUtils.get_directory_pdb_file_paths(des_dir.design_pdbs)
    # Make mutations to design sequences and gather mutations alone for residue specific processing
    # all_mutations = design_mutations_for_metrics(des_dir, wild_type_file=wild_type_file)
    # sequence_mutations = design_mutations_for_sequence(des_dir, wild_type_file)
    sequence_mutations = Ams.generate_mutations(all_design_files, wild_type_file)
    offset_dict = SDUtils.pdb_to_pose_num(sequence_mutations['ref'])
    # logger.debug('Chain offset: %s' % str(offset_dict))
    sequence_mutations.pop('ref')
    all_design_sequences = Ams.generate_sequences(wt_sequence, sequence_mutations)
    # logger.debug('All Sequences: %s' % all_design_sequences)
    for chain in all_design_sequences:
        all_design_sequences[chain] = remove_pdb_prefixes(all_design_sequences[chain])

    all_mutations = Ams.generate_mutations(all_design_files, wild_type_file, pose_num=True)
    all_mutations_no_chains = Ams.make_mutations_chain_agnostic(all_mutations)
    all_mutations_simplified = Ams.simplify_mutation_dict(all_mutations_no_chains)
    cleaned_mutations = remove_pdb_prefixes(all_mutations_simplified)

    # Process Residue and H-bond dataframe metrics
    # logger.debug('Design Scores: %s' % str(all_design_scores))
    interface_hbonds = hbond_processing(all_design_scores, hbonds_columns)
    number_hbonds = {}
    for entry in interface_hbonds:
        number_hbonds[entry] = len(interface_hbonds[entry])
    number_hbonds_df = pd.DataFrame(number_hbonds, index=['number_hbonds', ]).T
    scores_df = pd.merge(scores_df, number_hbonds_df, left_index=True, right_index=True)

    residue_dict = residue_processing(all_design_scores, cleaned_mutations, offset_dict, hbonds=interface_hbonds)
    res_df = pd.concat({key: pd.DataFrame(value) for key, value in residue_dict.items()}).unstack()

    # Subtract residue info from reference (refine)
    if delta_refine:
        # TODO Refine is not perfect reference in deltaG as modelling occurred currently only subtracting energy
        res_df.update(res_df.iloc[:, res_df.columns.get_level_values(1) == 'energy'].sub(
            res_df.loc[PUtils.stage[1], res_df.columns.get_level_values(1) == 'energy']))  # [res_columns_subtract]
        # df = df.sub(df.loc[PUtils.stage[1], df.columns.get_level_values(1)=='energy'])  # [res_columns_subtract]

    # Get unique protocols for protocol specific design metrics
    # unique_protocols = scores_df.columns.get_level_values(1)=='protocol'
    unique_protocols = strings_df['protocol'].unique().tolist()
    logger.debug('Unique Protocols: %s' % str(unique_protocols))
    # Drop protocol values
    try:
        unique_protocols.remove('')  # TODO remove after P432 MinMatch6 upon future script deployment
        unique_protocols.remove('refine')
    except ValueError:
        pass

    protocol_specific_designs = {}
    protocol_specific_sequences = {}
    per_res_keys = ['jsd', 'int_jsd']
    protocol_specific_stats = {protocol: {} for protocol in unique_protocols}
    for protocol in unique_protocols:
        protocol_specific_designs[protocol] = strings_df.index[strings_df['protocol'] == protocol].tolist()
        protocol_specific_sequences[protocol] = {chain: {name: all_design_sequences[chain][name]
                                                         for name in all_design_sequences[chain]
                                                         if name in protocol_specific_designs[protocol]}
                                                 for chain in all_design_sequences}
        # logger.debug(protocol_specific_sequences[protocol])
        res_profile = Ams.analyze_mutations(des_dir, protocol_specific_sequences[protocol])
        # protocol_specific_stats[protocol] = {}
        for key in per_res_keys:
            protocol_specific_stats[protocol][key + '_per_res'] = per_res_metric(res_profile, key=key)
            # protocol_specific_stats[protocol] = {'res_profile':
            #                                      Ams.analyze_mutations(des_dir, protocol_specific_sequences[protocol]}
    #       {protocol: {'res_profile': {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412},
    #       'jsd_per_res': 0.747, 'int_jsd_per_res': 0.412}, ...}, ...}

    # Prepare/Merge processed dataframes. First, make all multiindex columns
    strings_df.columns = pd.MultiIndex.from_product([['protocol'], strings_df.columns])
    scores_df.columns = pd.MultiIndex.from_product([['pose'], scores_df.columns])
    res_df = pd.merge(strings_df, res_df, left_index=True, right_index=True)
    scores_df = pd.merge(strings_df, scores_df, left_index=True, right_index=True)
    if merge_residue_data:
        scores_df = pd.merge(scores_df, res_df, left_index=True, right_index=True)  # , how='inner', on='index',

    # Drop refine row
    scores_df = scores_df.drop(PUtils.stage[1], axis=0)
    res_df = res_df.drop(PUtils.stage[1], axis=0)

    # Add pose metric statistics to total dataframe
    scores_df = scores_df.append(scores_df.mean().rename('average'))
    scores_df = scores_df.append(scores_df.std().rename('std_dev'))
    protocol_mean_df = scores_df.groupby(('protocol', 'protocol')).mean()
    protocol_std_df = scores_df.groupby(('protocol', 'protocol')).std()
    # scores_df = scores_df.append(scores_df.groupby(('protocol', 'protocol')).mean())
    protocol_std_df.index = protocol_std_df.index.to_series().map(
        {protocol: protocol + '_std_dev' for protocol in sorted(unique_protocols)})
    scores_df = scores_df.append(protocol_mean_df)
    scores_df = scores_df.append(protocol_std_df)  # .rename(str(protocol)))

    # POSE ANALYSIS: Create pose_average_df
    pose_average_df = pd.DataFrame(scores_df.loc['average', :]).T
    pose_average_df.index = [str(des_dir), ]

    # Add pose specific metrics to pose_average_df
    other_pose_metrics = gather_fragment_metrics(des_dir)
    int_residues = SDUtils.get_interface_residues(SDUtils.parse_design_flags(des_dir.path))
    other_pose_metrics['total_interface_residues'] = len(int_residues)
    wt_pdb = SDUtils.read_pdb(wild_type_file)
    chain_sep = wt_pdb.getTermCAAtom('C',
                                     wt_pdb.chain_id_list[0]).residue_number  # this only works with two chains TODO
    # TODO make sure clean_asu.pdb has B-factors included
    int_b_factor = 0
    for residue in int_residues:
        if residue <= chain_sep:
            int_b_factor += wt_pdb.get_ave_residue_b_factor(wt_pdb.chain_id_list[0], residue)
        else:
            int_b_factor += wt_pdb.get_ave_residue_b_factor(wt_pdb.chain_id_list[1], residue)
    other_pose_metrics['interface_b_factor_per_res'] = round(int_b_factor / len(int_residues), 2)

    fragment_metrics_df = pd.DataFrame(other_pose_metrics, index=[str(des_dir), ])
    fragment_metrics_df.columns = pd.MultiIndex.from_product([['pose'], fragment_metrics_df.columns])
    pose_average_df = pd.merge(pose_average_df, fragment_metrics_df, left_index=True, right_index=True)

    # Add Protocol specific metrics to pose_average_df
    for protocol in unique_protocols:
        temp_df = pd.DataFrame(scores_df.loc[protocol, (slice(None), specific_columns_to_add)]).T
        temp_df.index = [str(des_dir), ]
        temp_df.columns = temp_df.columns.droplevel(0)  # get rid of multi-index

        # From protocol dependent sequence stats, add stats as columns
        for stat in protocol_specific_stats[protocol]:
            temp_df[stat] = protocol_specific_stats[protocol][stat]

        temp_df.columns = pd.MultiIndex.from_product([[protocol], temp_df.columns])
        pose_average_df = pd.merge(pose_average_df, temp_df, left_index=True, right_index=True)

    # Remove pose specific metrics from pose_average_df
    remove_average_columns = ['protocol', ]
    pose_average_df = pose_average_df.drop(remove_average_columns, level=1, axis=1)

    # Save Pose, Residue dataframes
    _path = os.path.join(des_dir.all_scores, str(des_dir))
    trajectory_outpath = _path + '_Trajectories.csv'
    residue_outpath = _path + '_Residues.csv'
    scores_df.to_csv(trajectory_outpath)
    res_df.to_csv(residue_outpath)

    return pose_average_df

    # # ___________________________________________________________
    #
    # all_design_scores = read_scores(os.path.join(des_dir.scores, PUtils.scores_file))
    #
    # # all_design_scores['refine'] = refine_score[0].pop()
    # scores_df = pd.DataFrame.from_dict(all_design_scores, orient='index')
    # scores_df = scores_df.rename(columns=rename_columns)
    # scores_df = scores_df.groupby(level=0, axis=1).apply(lambda x: x.apply(join_columns, axis=1))
    # # refine_score_df = scores(os.path.join(des_dir.scores, 'score_%s.sc' % PUtils.stage[1]))
    # # refine_score_df = refine_score_df.rename(index={'0001': 'Refine'})
    # # scores_df = scores(os.path.join(des_dir.scores, 'score_%s.sc' % PUtils.stage[2]))
    # # scores_df = scores_df.sub(refine_score_df.loc['Refine', ])
    # scores_df.append(scores_df.mean().rename('total_average'))
    # scores_df.append(scores_df.std().rename('total_std_dev'))
    # for protocol in scores_df['protocol'].unique():
    #     scores_df.append(scores_df[scores_df['protocol'] == protocol].mean().rename(protocol + '_average'))
    #     scores_df.append(scores_df[scores_df['protocol'] == protocol].std().rename(protocol + '_std_dev'))  # scores_df=
    # # scores_df.loc['Average', ] = scores_df.loc['Average', ].add(refine_score_df.loc['Refine', ])
    # # scores_df = scores_df.append(refine_score_df.loc['Refine', ])
    #
    # # Find all mutations made in the design sequences
    # all_mutations = design_mutations_for_metrics(des_dir, wildtype_file=Ams.get_wildtype_file(des_dir))
    # all_mutations_no_chains = Ams.make_mutations_chain_agnostic(all_mutations)
    # all_mutations_simplified = Ams.simplify_mutation_dict(all_mutations_no_chains)
    #
    # # Combine: all mutations key is identical to key in all_design_scores (decoy names). chain_id removed
    # all_design_scores.update(all_mutations_simplified)
    #
    # wt_sequence = Ams.get_pdb_sequences(wild_type_file)
    # all_design_sequences = Ams.generate_sequences(wt_sequence, all_mutations)
    # # all_design_sequences = Ams.mutate_wildtype_sequences(all_design_files, wild_type_file)
    sequence_analysis = Ams.analyze_mutations(des_dir.path, all_design_sequences, print_results=print_output)
    # # ex: {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732}, ...}

    if print_output:
        print(sequence_analysis)
        print(scores_df)
    else:
        # Write sequence analysis and score analysis to files
        scores_df.to_csv(os.path.join(des_dir.all_scores, str(des_dir) +
                                      os.path.splitext(PUtils.scores_file)[0] + '.csv'))
        with open(os.path.join(des_dir.all_scores, str(des_dir) + '_sequence.pkl'), 'wb') as f:
            pickle.dump(sequence_analysis, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='%s\nAnalyze the output of %s for design curation. Requires a '
                                                 'directory with wild-type PDB/PSSM/flags_design files, a folder of '
                                                 'design PDB files, and a score.sc file with json objects as input.'
                                                 % (__name__, PUtils.program_name))
    parser.add_argument('-d', '--directory', type=str, help='Directory where Nanohedra output is located. Default=CWD',
                        default=os.getcwd())
    parser.add_argument('-f', '--file', type=str, help='File with location(s) of Nanohedra output.', default=None)
    parser.add_argument('-m', '--multi_processing', action='store_true', help='Should job be run with multiprocessing? '
                                                                              'Default=False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out? Default=False')
    parser.add_argument('-j', '--join', action='store_true', help='Join Trajectory and Residue Dataframes?'
                                                                   ' Default=False')
    parser.add_argument('-g', '--delta_g', action='store_true', help='Compute deltaG versus Refine structure? '
                                                                     'Default=False')
    args = parser.parse_args()

    # Start logging output
    if args.debug:
        logger = SDUtils.start_log(name='main', level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = SDUtils.start_log(name='main', level=2)

    logger.info('Starting %s with options:\n%s' %
                (__name__, '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    # Collect all designs to be processed
    all_designs, location = SDUtils.collect_designs(args.directory, file=args.file)
    assert all_designs != list(), logger.critical('No %s directories found within \'%s\' input! Please ensure correct '
                                                  'location.' % (PUtils.nano, location))
    logger.info('%d total Poses found in \'%s\'.' % (len(all_designs), location))
    logger.info('All pose specific logs are located in their corresponding directories. For example \'%s\'' %
                os.path.join(all_designs[0].path, os.path.basename(all_designs[0].path) + '.log'))

    # Start Pose processing
    if args.multi_processing:
        # Calculate the number of threads to use depending on computer resources
        mp_threads = SDUtils.calculate_mp_threads(args.suspend)
        logger.info('Beginning Pose specific multiprocessing with %s multiprocessing threads' % str(mp_threads))
        zipped_args = zip(all_designs, repeat(args.delta_g), repeat(args.join), repeat(args.debug))
        pose_results, exceptions = SDUtils.mp_starmap(analyze_output, zipped_args, mp_threads)
        if exceptions:
            logger.warning('\nThe following exceptions were thrown. Design for these directories is inaccurate.')
            for exception in exceptions:
                logger.warning(exception)
    else:
        logger.info('Beginning Pose specific processing. If single process is taking a while, use -m during submission')
        pose_results = []
        for des_directory in all_designs:
            pose_results.append(analyze_output(des_directory, delta_refine=args.delta_g, merge_residue_data=args.join,
                                               debug=args.debug))

    design_df = pd.DataFrame()
    for result in pose_results:
        design_df = design_df.append(result, ignore_index=True)

    # Save Design dataframe
    out_path = os.path.join(args.directory, 'AllDesignPoseMetrics.csv')
    design_df.to_csv(out_path)
    logger.info('All Design Poses were written to %s' % out_path)
