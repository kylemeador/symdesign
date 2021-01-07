import argparse
import copy
import math
import operator
import os
from itertools import repeat, combinations
from json import loads

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import DesignDirectory
import PathUtils as PUtils
# import PDB
import SequenceProfile
import SymDesignUtils as SDUtils

# import CmdUtils as CUtils


# Globals
groups = 'protocol'
metric_master = {'average_fragment_z_score': 'The average z-score per fragment used in docking/redesign',
                 'buns_heavy_total': 'Buried unsaturated Hbonding heavy atoms in the pose',
                 'buns_hpol_total': 'Buried unsaturated Hbonding hydrogen atoms in the pose',
                 'buns_total': 'Total buried unsaturated Hbonds in the pose',
                 'buns_per_ang': 'Buried Unsaturated Hbonds per Angstrom^2 of interface',
                 'contact_count': 'Number of carbon-carbon contacts across interface',
                 'core': 'The number of core residues as classified by E. Levy 2010',
                 'cst_weight': 'Total weight of coordinate constraints to keep pose stationary in 3D space',
                 'divergence_combined_per_residue': 'The Jensen-Shannon divergence of interface residues from the'
                                                    'position specific combined (fragment & evolution) values',
                 'divergence_fragment_per_residue': 'The Jensen-Shannon divergence of interface residues from the'
                                                    'position specific fragment profile values',
                 'divergence_evolution_per_residue': 'The Jensen-Shannon divergence of interface residues from the'
                                                     'position specific evolutionary profile values',
                 'divergence_interface_per_residue': 'The Jensen-Shannon divergence of interface residues from the'
                                                     'interface DB background values',
                 'fsp_energy': 'Total weight of sequence constraints used to favor certain amino acids in design. '
                               'Only some protocols have values',
                 'fsp_total_stability': 'fsp_energy + total pose energy',  # DEPRECIATED
                 'full_stability_complex': 'Total pose energy (essentially REU)',  # DEPRECIATED
                 'full_stability_A_oligomer': 'Total A oligomer pose energy (essentially REU)',  # DEPRECIATED
                 'full_stability_B_oligomer': 'Total B oligomer pose energy (essentially REU)',  # DEPRECIATED
                 'full_stability_oligomer': 'Total oligomer pose energy (essentially REU)',  # DEPRECIATED
                 'int_area_hydrophobic': 'Total interface buried surface area hydrophobic',
                 'int_area_polar': 'Total interface buried surface area polar',
                 'int_area_res_summary_hydrophobic_A_oligomer': 'Sum of each interface residue\'s hydrophobic area for '
                                                                'oligomer A',
                 'int_area_res_summary_hydrophobic_B_oligomer': 'Sum of each interface residue\'s hydrophobic area for '
                                                                'oligomer B',
                 'int_area_res_summary_polar_A_oligomer': 'Sum of each interface residue\'s polar area for oligomer A',
                 'int_area_res_summary_polar_B_oligomer': 'Sum of each interface residue\'s polar area for oligomer B',
                 'int_area_res_summary_total_A_oligomer': 'Sum of each interface residue\'s total area for oligomer A',
                 'int_area_res_summary_total_B_oligomer': 'Sum of each interface residue\'s total area for oligomer B',
                 'int_area_total': 'Total interface solvent accessible surface area',
                 'int_composition_diff': 'The similarity to the expected interface composition given BSA. 1 is similar',
                 'int_connectivity_A': 'Interface connection chainA to the rest of the protein',
                 'int_connectivity_B': 'Interface connection chainB to the rest of the protein',
                 'int_energy_context_A_oligomer': 'Interface energy of the A oligomer',  # DEPRECIATED x2
                 'int_energy_context_B_oligomer': 'Interface energy of the B oligomer',  # DEPRECIATED x2
                 'int_energy_context_oligomer_A': 'Interface energy of the A oligomer',  # DEPRECIATED
                 'int_energy_context_oligomer_B': 'Interface energy of the B oligomer',  # DEPRECIATED
                 'int_energy_context_complex': 'interface energy of the complex',  # DEPRECIATED
                 'int_energy_res_summary_A_oligomer': 'Sum of each interface residue\'s energy for oligomer A',  # DEPRECIATED
                 'int_energy_res_summary_B_oligomer': 'Sum of each interface residue\'s energy for oligomer B',  # DEPRECIATED
                 'int_energy_res_summary_complex': 'Sum of each interface residue\'s energy for the complex',
                 'int_energy_res_summary_delta': 'Delta of int_energy_res_summary_complex and _oligomer',
                 'int_energy_res_summary_oligomer': 'Sum of each interface residue\'s energy for total oligomer',
                 'int_energy_res_summary_oligomer_A': 'Sum of each interface residue\'s energy for oligomer A',
                 'int_energy_res_summary_oligomer_B': 'Sum of each interface residue\'s energy for oligomer B',
                 'int_separation': 'Median distance between all atom points on two sides of a interface (SC term)',
                 'interaction_energy_complex': 'The two-body interface energy of the assembled complex',
                 'interface_b_factor_per_res': 'The average interface residue B-factor as measure from the PDB',
                 'shape_complementarity': 'Interface shape complementarity (SC). Measure of fit between two surfaces',
                 'number_hbonds': 'The number of Hbonding residues present in the interface',
                 'observations': 'Number of unique data points',
                 'observed_combined': 'Percent of observed residues in combined profile',
                 'observed_evolution': 'Percent of observed residues in evolutionary profile',
                 'observed_interface': 'Percent of observed residues in fragment profile',
                 'percent_core': 'The percentage of total residues which are \'core\' according to Levy, E. 2010',
                 'percent_fragment': 'Percent of residues with fragment data out of total residues',
                 'percent_int_area_hydrophobic': 'The percent of interface area which is occupied by hydrophobic atoms',
                 'percent_int_area_polar': 'The percent of interface area which is occupied by polar atoms',
                 'percent_rim': 'The percentage of total residues which are \'rim\' according to Levy, E. 2010',
                 'percent_support': 'The percentage of total residues which are \'support\' according to Levy, E. 2010',
                 'protocol_energy_distance_sum': 'The distance between the average linearly embedded per residue energy'
                                                 ' covariation between specified protocols. Larger = greater distance',
                 'protocol_similarity_sum': 'The similarity between specified protocols. Larger is more similar',
                 'protocol_seq_distance_sum': 'The distance between the average linearly embedded sequence differences '
                                              'between specified protocols. Larger = greater distance',
                 'ref': 'Rosetta Energy Term - A metric for the unfolded protein energy and some sequence fitting '
                        'corrections to the score function',
                 'rim': 'The number of rim residues as classified by E. Levy 2010',
                 'rmsd': 'Root Mean Square Deviation of all CA atoms between relaxed and design state',
                 groups: 'Protocols I to search sequence space for fragments and evolutionary information',
                 'solvation_energy': 'Energy required to hydrate the unbound components.',
                 'support': 'The number of support residues as classified by E. Levy 2010',
                 'symmetry': 'The specific symmetry type used design (point, layer, lattice)',
                 'nanohedra_score': 'Sum of the inverse of each fragments Z-score capped at 3 = 1 / Z-score (maximum3)', # DEPRECIATED
                 # 'nanohedra_score': 'Sum of the inverse of each fragments Z-score capped at 3 = 1 / Z-score (maximum3)',
                 'fragment_z_score_total': 'The sum of all fragments Z-Scores',
                 'unique_fragments': 'The number of unique fragments placed on the pose',
                 'total_interface_residues': 'The number of interface residues found in the pose',
                 'interface_b_factor_per_res': 'The average B factor for each atom in each interface residue',
                 'REU': 'Rosetta Energy Units. Always 0. We can disregard',
                 'buns_asu': 'Buried unsaturated hydrogen bonds. This column helps with buns_total',  # DEPRECIATED
                 'buns_asu_hpol': 'Buried unsaturated hydrogen bonds. This column helps with buns_total',  # DEPRECIATED
                 'buns_nano': 'Buried unsaturated hydrogen bonds. This column helps with buns_total',  # DEPRECIATED
                 'buns_nano_hpol': 'Buried unsaturated hydrogen bonds. This column helps with buns_total',  # DEPRECIATED
                 'int_area_asu_hydrophobic': 'Buried surface area in asu interface hydrophobic',  # UNUSED
                 'int_area_asu_polar': 'Buried surface area in asu interface area polar',  # UNUSED
                 'int_area_asu_total': 'Buried surface area in asu interface area total',  # UNUSED
                 'int_area_ex_asu_hydrophobic': 'Buried surface area in extra-asu interface area hydrophobic',  # UNUSED
                 'int_area_ex_asu_polar': 'Buried surface area in extra-asu interface area polar',  # UNUSED
                 'int_area_ex_asu_total': 'Buried surface area in extra-asu interface area total',  # UNUSED
                 'int_connectivity1': 'Old connectivity',  # DEPRECIATED
                 'int_connectivity2': 'Old connectivity',  # DEPRECIATED
                 'int_energy_context_asu': 'Interface energy of the ASU',  # UNUSED, DEPRECIATED
                 'int_energy_context_unbound': 'Interface energy of the unbound complex',  # UNUSED, DEPRECIATED
                 'coordinate_constraint': 'Same as cst_weight',
                 'int_energy_res_summary_asu': 'Sum of each interface residues individual energy for the ASU',  # DEPRECIATED
                 'int_energy_res_summary_unbound': 'Sum of each interface residues individual energy for the unbound',  # DEPRECIATED
                 'interaction_energy': 'Interaction energy between two sets of residues',  # DEPRECIATED
                 'interaction_energy_asu': 'Interaction energy between two sets of residues in ASU state',  # DEPRECIATED
                 'interaction_energy_oligomerA': 'Interaction energy between two sets of residues in oligomerA',  # DEPRECIATED
                 'interaction_energy_oligomerB': 'Interaction energy between two sets of residues in oligomerB',  # DEPRECIATED
                 'interaction_energy_unbound': 'Interaction energy between two sets of residues in unbound state', # DEPRECIATED
                 'res_type_constraint': 'Same as fsp_energy',
                 'time': 'Time for the protocol to complete',
                 'hbonds_res_selection_complex': 'The specific hbonds present in the bound pose',
                 'hbonds_res_selection_A_oligomer': 'The specific hbonds present in the oligomeric pose A',
                 'hbonds_res_selection_B_oligomer': 'The specific hbonds present in the oligomeric pose B',
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
                 'hbond_sc': 'Rosetta Energy Term - side-chain hydrogen bonding',
                 'hbond_sr_bb': 'Rosetta Energy Term - short range backbone hydrogen bonding',
                 'lk_ball_wtd': 'Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
                 'omega': 'Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
                 'p_aa_pp': '"Rosetta Energy Term - statistical probability of an amino acid given angles phi',
                 'pro_close': 'Rosetta Energy Term - to favor closing of proline rings',
                 'rama_prepro': 'Rosetta Energy Term - amino acid dependent term to favor certain ramachandran angles '
                                'on residue before prolines',
                 'yhh_planarity': 'Rosetta Energy Term - to favor planarity of tyrosine hydrogen'}

# These metrics are necessary for all calculations performed during the analysis script. If missing, something will fail
necessary_metrics = {'buns_asu_hpol', 'buns_nano_hpol', 'buns_asu', 'buns_nano', 'buns_total', 'contact_count',
                     'cst_weight', 'fsp_energy', 'int_area_hydrophobic', 'int_area_polar', 'int_area_total',
                     'int_connectivity_A', 'int_connectivity_B', 'int_energy_context_oligomer_A',
                     'int_energy_context_oligomer_B', 'int_energy_context_complex', 'int_energy_res_summary_oligomer_A',
                     'int_energy_res_summary_oligomer_B', 'int_energy_res_summary_complex', 'int_separation',
                     'interaction_energy_complex', groups, 'ref', 'rmsd', 'shape_complementarity', 'symmetry_switch',
                     'hbonds_res_selection_complex', 'hbonds_res_selection_A_oligomer',
                     'hbonds_res_selection_B_oligomer'}

#                      'fsp_total_stability', 'full_stability_complex',
#                      'int_area_res_summary_hydrophobic_A_oligomer', 'int_area_res_summary_hydrophobic_B_oligomer',
#                      'int_area_res_summary_polar_A_oligomer', 'int_area_res_summary_polar_B_oligomer',
#                      'int_area_res_summary_total_A_oligomer', 'int_area_res_summary_total_B_oligomer',
#                      'int_energy_res_summary_delta', 'number_hbonds', 'total_interface_residues',
#                      'int_energy_context_delta',
#                      'average_fragment_z_score', 'nanohedra_score', 'unique_fragments', 'interface_b_factor_per_res',
#                      'int_energy_res_summary_oligomer', 'int_energy_context_oligomer',

final_metrics = {'buns_heavy_total', 'buns_hpol_total', 'buns_total', 'contact_count', 'cst_weight', 'fsp_energy',
                 'percent_fragment', 'int_area_hydrophobic', 'int_area_polar', 'int_area_total', 'int_connectivity_A',
                 'int_connectivity_B', 'int_energy_res_summary_A_oligomer', 'int_energy_res_summary_B_oligomer',
                 'int_energy_res_summary_complex', 'int_energy_res_summary_delta', 'int_energy_res_summary_oligomer',
                 'int_separation', 'interaction_energy_complex', 'number_hbonds', 'observed_combined',
                 'observed_evolution', 'observed_interface', 'ref', 'rmsd', 'shape_complementarity', 'solvation_energy'}
#               These are missing the bb_hb contribution and are inaccurate
#                  'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex',
#                  'int_energy_context_delta', 'int_energy_context_oligomer',
#               These are accounted for in other pose metrics
#                  'nanohedra_score', 'average_fragment_z_score', 'unique_fragments', 'total_interface_residues',
#                  'interface_b_factor_per_res'}
#               These could be added in, but seem to be unnecessary
#                  'fsp_total_stability', 'full_stability_complex',
#                  'int_area_res_summary_hydrophobic_A_oligomer', 'int_area_res_summary_hydrophobic_B_oligomer',
#                  'int_area_res_summary_polar_A_oligomer', 'int_area_res_summary_polar_B_oligomer',
#                  'int_area_res_summary_total_A_oligomer', 'int_area_res_summary_total_B_oligomer',

columns_to_rename = {'int_sc': 'shape_complementarity', 'int_sc_int_area': 'int_area',
                     'int_sc_median_dist': 'int_separation', 'R_full_stability': 'R_full_stability_complex',
                     'int_energy_context_A_oligomer': 'int_energy_context_oligomer_A',
                     'int_energy_context_B_oligomer': 'int_energy_context_oligomer_B',
                     'int_energy_res_summary_B_oligomer': 'int_energy_res_summary_oligomer_B',
                     'int_energy_res_summary_A_oligomer': 'int_energy_res_summary_oligomer_A',
                     'relax_switch': groups, 'no_constraint_switch': groups,
                     'limit_to_profile_switch': groups, 'combo_profile_switch': groups,
                     'favor_profile_switch': groups, 'consensus_design_switch': groups}
#                      'total_score': 'REU', 'decoy': 'design', 'symmetry_switch': 'symmetry',

columns_to_remove = ['decoy', 'symmetry_switch', 'oligomer_switch', 'total_score',
                     'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex']
remove_score_columns = ['hbonds_res_selection_asu', 'hbonds_res_selection_unbound']

# sum columns using tuple [0] + [1]
summation_pairs = {'buns_hpol_total': ('buns_asu_hpol', 'buns_nano_hpol'),
                   'buns_heavy_total': ('buns_asu', 'buns_nano'),
                   # 'int_energy_context_oligomer':
                   #     ('int_energy_context_oligomer_A', 'int_energy_context_oligomer_B'),
                   'int_energy_res_summary_oligomer':
                       ('int_energy_res_summary_oligomer_A', 'int_energy_res_summary_oligomer_B')}  # ,
#                    'full_stability_oligomer': ('full_stability_A_oligomer', 'full_stability_B_oligomer')}  # ,
#                    'hbonds_oligomer': ('hbonds_res_selection_A_oligomer', 'hbonds_res_selection_B_oligomer')}

# subtract columns using tuple [0] - [1] to make delta column
delta_pairs = {'int_energy_res_summary_delta': ('int_energy_res_summary_complex', 'int_energy_res_summary_oligomer'),
               'solvation_energy': ('interaction_energy_complex', 'int_energy_res_summary_delta')}
#                'int_energy_context_delta': ('int_energy_context_complex', 'int_energy_context_oligomer'),
#                'full_stability_delta': ('full_stability_complex', 'full_stability_oligomer')}
#                'number_hbonds': ('hbonds_res_selection_complex', 'hbonds_oligomer')}
#               TODO P432 full_stability'_complex'

# divide columns using tuple [0] / [1] to make divide column
division_pairs = {'percent_int_area_hydrophobic': ('int_area_hydrophobic', 'int_area_total'),
                  'percent_int_area_polar': ('int_area_polar', 'int_area_total'),
                  'percent_core': ('core', 'total_interface_residues'),
                  'percent_rim': ('rim', 'total_interface_residues'),
                  'percent_support': ('support', 'total_interface_residues'),
                  'buns_per_ang': ('buns_total', 'int_area_total')}

# Some of these are unneeded now, but handing around incase renaming has occurred
unnecessary = ['int_area_asu_hydrophobic', 'int_area_asu_polar', 'int_area_asu_total',
               'int_area_ex_asu_hydrophobic', 'int_area_ex_asu_polar', 'int_area_ex_asu_total',
               'buns_asu', 'buns_asu_hpol', 'buns_nano', 'buns_nano_hpol', 'int_connectivity1',
               'int_connectivity2', 'int_energy_context_asu', 'int_energy_context_unbound',
               'coordinate_constraint', 'int_energy_res_summary_asu', 'int_energy_res_summary_unbound',
               'interaction_energy', 'interaction_energy_asu', 'interaction_energy_oligomerA',
               'interaction_energy_oligomerB', 'interaction_energy_unbound', 'res_type_constraint', 'time', 'REU',
               'full_stability_complex', 'full_stability_oligomer', 'fsp_total_stability',
               'int_area_res_summary_hydrophobic_A_oligomer', 'int_area_res_summary_hydrophobic_B_oligomer',
               'int_area_res_summary_polar_A_oligomer', 'int_area_res_summary_polar_B_oligomer',
               'int_area_res_summary_total_A_oligomer', 'int_area_res_summary_total_B_oligomer',
               'full_stability_A_oligomer', 'full_stability_B_oligomer',
               'full_stability_oligomer_A', 'full_stability_oligomer_B']  # TODO Remove once Entry65 is done

# All terms but ref as this seems useful to keep
rosetta_terms = ['lk_ball_wtd', 'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'yhh_planarity', 'dslf_fa13',
                 'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb']  # 'ref'

# Current protocols in use in design.xml
protocols = ['combo_profile_switch', 'favor_profile_switch', 'limit_to_profile_switch', 'no_constraint_switch']
protocols_of_interest = ['combo_profile', 'limit_to_profile', 'no_constraint']

# Specific columns of interest to distinguish between design trajectories
protocol_specific_columns = ['shape_complementarity', 'buns_total', 'contact_count', 'int_energy_res_summary_delta',
                             'int_area_total', 'number_hbonds', 'percent_int_area_hydrophobic',  # ]
                             'interaction_energy_complex']
#                              'int_area_hydrophobic', 'int_area_polar',
#                              'full_stability_delta', 'int_energy_context_delta',

stats_metrics = ['mean', 'std']
residue_classificiation = ['core', 'rim', 'support']  # 'hot_spot'
# per_res_keys = ['jsd', 'des_jsd', 'int_jsd', 'frag_jsd']

# from table 1, theoretical values of Tien et al. 2013
gxg_sasa = {'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167, 'E': 223, 'Q': 225, 'G': 104, 'H': 224, 'I': 197,
            'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159, 'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174}


def read_scores(file, key='decoy'):
    """Take a json formatted score.sc file and incorporate into dictionary of dictionaries with 'key' value as outer key

    Args:
        file (str): Location on disk of scorefile
    Keyword Args:
        key='decoy' (str): Name of the json key to use as outer dictionary identifier
    Returns:
        score_dict (dict): {design_name: {all_score_metric_keys: all_score_metric_values}, ...}
    """
    with open(file, 'r') as f:
        score_dict = {}
        for score in f.readlines():
            entry = loads(score)
            design = entry[key]  # entry['decoy'].split('_')[-1]
            if design not in score_dict:
                score_dict[design] = entry
            else:
                # to ensure old trajectories don't have lingering protocol info TODO clean Rosetta protocol generation
                for protocol in protocols:
                    if protocol in entry.keys():
                        for rm_protocol in protocols:
                            try:
                                score_dict[design].pop(rm_protocol)
                            except KeyError:
                                pass
                score_dict[design].update(entry)

    return score_dict


def remove_pdb_prefixes(pdb_dict):
    """Strip all but last key identifiers separated by '_' from keys of dictionary. Ex 'design_0001' -> '0001'"""
    clean_key_dict = {}
    for key in pdb_dict:
        new_key = key.split('_')[-1]
        clean_key_dict[new_key] = pdb_dict[key]

    return clean_key_dict


def join_columns(x):
    """Combine columns in a dataframe with the same column name. Keep only the last column record"""

    new_data = ','.join(x[x.notnull()].astype(str))
    return new_data.split(',')[-1]


def columns_to_new_column(df, column_dict, mode='add'):
    """Find new column values by taking an operation of one column on another

    Args:
        df (pandas.DataFrame): Dataframe where the columns are located
        column_dict (dict): A dictionary with keys as new column names, values as tuple of columns.
            Where value[0] mode(operation) value[1] = key
    Keyword Args:
        mode='add' (str) = What operator to use?
            Viable options are included in module operator, but could be 'sub', 'mul', 'truediv', etc.
    Returns:
        df (pandas.DataFrame): Dataframe with new column values
    """
    for column in column_dict:
        try:
            df[column] = operator.attrgetter(mode)(operator)(df[column_dict[column][0]], df[column_dict[column][1]])
        except KeyError:
            pass

    return df


def calc_relative_sa(aa, sa):  # TODO SDUtils
    """Calculate relative surface area according to theoretical values of Tien et al. 2013"""
    return round(sa / gxg_sasa[aa], 2)


def hbond_processing(score_dict, columns, offset=None):
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering
    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        columns (list): ['hbonds_res_selection_complex', 'hbonds_res_selection_A_oligomer',
            'hbonds_res_selection_B_oligomer']
    Keyword Args:
        offset=None (dict): {'A': 0, 'B': 102}
    Returns:
        hbond_dict (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    """
    hbond_dict = {}
    res_offset = 0
    for entry in score_dict:
        entry_dict = {}
        for column in columns:
            hbonds = score_dict[entry][column].split(',')
            if hbonds[0] == '' and len(hbonds) == 1:
                hbonds = set()
            else:
                if column.split('_')[-1] == 'oligomer' and offset:
                    res_offset = offset[column.split('_')[-2]]
                for i in range(len(hbonds)):
                    hbonds[i] = int(hbonds[i][:-1]) + res_offset  # remove chain ID off last index
            entry_dict[column.split('_')[3]] = set(hbonds)
        if len(entry_dict) == 3:
            hbond_dict[entry] = list((entry_dict['complex'] - entry_dict['A']) - entry_dict['B'])
        else:
            hbond_dict[entry] = list()
        #     logger.error('%s: Missing hbonds_res_selection_ data for %s. Hbonds inaccurate!' % (pose, entry))

    return hbond_dict


def dirty_hbond_processing(score_dict, offset=None):  # columns
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering
    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
    Keyword Args:
        offset=None (dict): {'A': 0, 'B': 102} The amount to offset each chain by
    Returns:
        hbond_dict (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    """
    hbond_dict = {}
    res_offset = 0
    for entry in score_dict:
        entry_dict = {}
        # for column in columns:
        for column in score_dict[entry]:
            if column.startswith('hbonds_res_selection'):
                hbonds = score_dict[entry][column].split(',')
                # ensure there are hbonds present
                if hbonds[0] == '' and len(hbonds) == 1:
                    hbonds = set()
                else:
                    if column.split('_')[-1] == 'oligomer' and offset:
                        # find offset according to chain
                        res_offset = offset[column.split('_')[-2]]
                    for i in range(len(hbonds)):
                        hbonds[i] = int(hbonds[i][:-1]) + res_offset  # remove chain ID off last index of string
                entry_dict[column.split('_')[3]] = set(hbonds)
        if len(entry_dict) == 3:
            hbond_dict[entry] = list((entry_dict['complex'] - entry_dict['A']) - entry_dict['B'])
        else:
            hbond_dict[entry] = list()
            # logger.error('%s: Missing hbonds_res_selection_ data for %s. Hbonds inaccurate!' % (pose, entry))

    return hbond_dict


def hot_spot(residue_dict, energy=-1.5):  # UNUSED
    """Calculate if each residue in a dictionary is a hot-spot

    Args:
        residue_dict (dict)
    Keyword Args:
        energy=-1.5 (float): The threshold for hot spot consideration
    Returns:
        residue_dict (dict):
    """
    for res in residue_dict:
        if residue_dict[res]['energy'] <= energy:
            residue_dict[res]['hot_spot'] = 1
        else:
            residue_dict[res]['hot_spot'] = 0

    return residue_dict


def residue_composition_diff(row):
    """Calculate the composition difference for pose residue classification

    Args:
        row (pandas.Series): Series with 'int_area_total', 'core', 'rim', and 'support' indices
    Returns:
        (float): Difference of expected residue classification and observed
    """
    # Calculate modelled number of residues according to buried surface area (Levy, E 2010)
    def core_res_fn(bsa):
        return 0.01 * bsa + 0.6

    def rim_res_fn(bsa):
        return 0.01 * bsa - 2.5

    def support_res_fn(bsa):
        return 0.006 * bsa + 5

    classification_fxn_d = {'core': core_res_fn, 'rim': rim_res_fn, 'support': support_res_fn}
    class_ratio_diff_d = {}
    int_area = row['int_area_total']  # buried surface area
    if int_area <= 250:
        return np.nan
    #     assert int_area > 250, 'int_area_total gives negative value for support'

    for _class in classification_fxn_d:
        expected = classification_fxn_d[_class](int_area)
        class_ratio_diff_d[_class] = (1 - (abs(row[_class] - expected) / expected))
        if class_ratio_diff_d[_class] < 0:
            # above calculation fails to bound between 0 and 1 with large obs values due to proportion > 1
            class_ratio_diff_d[_class] = 0
    _sum = 0
    for value in class_ratio_diff_d.values():
        _sum += value

    return _sum / 3.0


def residue_processing(score_dict, mutations, columns, offset=None, hbonds=None):
    """Process Residue Metrics from Rosetta score dictionary

    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        mutations (dict): {'0001': {mutation_index: {'from': 'A', 'to: 'K'}, ...}, ...}
        columns (list): ['per_res_energy_complex_5', 'per_res_sasa_polar_A_oligomer_5', 
            'per_res_energy_A_oligomer_5', ...]
    Keyword Args:
        offset=None (dict): {'A': 0, 'B': 102}
        hbonds=None (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    Returns:
        residue_dict (dict): {'0001': {15: {'type': 'T', 'energy_delta': -2.771, 'bsa_polar': 13.987, 'bsa_hydrophobic': 
            22.29, 'bsa_total': 36.278, 'hbond': 0, 'core': 0, 'rim': 1, 'support': 0}, ...}, ...}
    """  # , 'hot_spot': 1
    dict_template = {'energy': {'complex': 0, 'oligomer': 0, 'fsp': 0, 'cst': 0},
                     'sasa': {'polar': {'complex': 0, 'oligomer': 0}, 'hydrophobic': {'complex': 0, 'oligomer': 0},
                              'total': {'complex': 0, 'oligomer': 0}},
                     'type': None, 'hbond': 0, 'core': 0, 'interior': 0, 'rim': 0, 'support': 0}  # , 'hot_spot': 0}
    total_residue_dict = {}
    for entry in score_dict:
        residue_dict = {}
        # for key, value in score_dict[entry].items():
        for column in columns:
            metadata = column.split('_')
            # if key.startswith('per_res_'):
            # metadata = key.split('_')
            res = int(metadata[-1])
            r_type = metadata[2]  # energy or sasa
            pose_state = metadata[-2]  # oligomer or complex or cst (constraint) or fsp (favor_sequence_profile)
            if pose_state == 'oligomer' and offset:
                res += offset[metadata[-3]]  # get oligomer chain length offset
            if res not in residue_dict:
                residue_dict[res] = copy.deepcopy(dict_template)
            if r_type == 'sasa':
                # Ex. per_res_sasa_hydrophobic_A_oligomer_15 or per_res_sasa_hydrophobic_complex_15
                polarity = metadata[3]  # hydrophobic or polar or total
                residue_dict[res][r_type][polarity][pose_state] = round(score_dict[entry][column], 3)
            else:
                # Ex. per_res_energy_A_oligomer_15 or per_res_energy_complex_15
                residue_dict[res][r_type][pose_state] = round(score_dict[entry][column], 3)
        # if residue_dict:
        for res in residue_dict:
            try:
                residue_dict[res]['type'] = mutations[entry][res]
            except KeyError:
                residue_dict[res]['type'] = mutations['ref'][res]  # fill with aa from wild_type sequence
            if hbonds:
                if res in hbonds[entry]:
                    residue_dict[res]['hbond'] = 1
            residue_dict[res]['energy_delta'] = residue_dict[res]['energy']['complex'] \
                - residue_dict[res]['energy']['oligomer'] - residue_dict[res]['energy']['cst']
            # - residue_dict[res]['energy']['fsp']
            rel_oligomer_sasa = calc_relative_sa(residue_dict[res]['type'],
                                                 residue_dict[res]['sasa']['total']['oligomer'])
            rel_complex_sasa = calc_relative_sa(residue_dict[res]['type'],
                                                residue_dict[res]['sasa']['total']['complex'])
            for polarity in residue_dict[res]['sasa']:
                # convert sasa measurements into bsa measurements
                residue_dict[res]['bsa_' + polarity] = round(
                    residue_dict[res]['sasa'][polarity]['oligomer'] - residue_dict[res]['sasa'][polarity][
                        'complex'], 2)
            if residue_dict[res]['bsa_total'] > 0:
                if rel_oligomer_sasa < 0.25:
                    residue_dict[res]['support'] = 1
                elif rel_complex_sasa < 0.25:
                    residue_dict[res]['core'] = 1
                else:
                    residue_dict[res]['rim'] = 1
            else:  # remove residue from dictionary as no interface design should be done? keep interior residues
                if rel_complex_sasa < 0.25:
                    residue_dict[res]['interior'] = 1
                # else:
                #     residue_dict[res]['surface'] = 1

            residue_dict[res].pop('sasa')
            residue_dict[res].pop('energy')
            # if residue_dict[res]['energy'] <= hot_spot_energy:
            #     residue_dict[res]['hot_spot'] = 1
        total_residue_dict[entry] = residue_dict

    return total_residue_dict


def dirty_residue_processing(score_dict, mutations, offset=None, hbonds=None):
    """Process Residue Metrics from Rosetta score dictionary

    One-indexed residues
    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        mutations (dict): {'0001': {mutation_index: {'from': 'A', 'to: 'K'}, ...}, ...}
    Keyword Args:
        offset=None (dict): {'A': 0, 'B': 102}
        hbonds=None (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    Returns:
        residue_dict (dict): {'0001': {15: {'type': 'T', 'energy_delta': -2.771, 'bsa_polar': 13.987, 'bsa_hydrophobic': 
            22.29, 'bsa_total': 36.278, 'hbond': 0, 'core': 0, 'rim': 1, 'support': 0}, ...}, ...}  # , 'hot_spot': 1
    """
    dict_template = {'energy': {'complex': 0, 'oligomer': 0, 'fsp': 0, 'cst': 0},
                     'sasa': {'polar': {'complex': 0, 'oligomer': 0}, 'hydrophobic': {'complex': 0, 'oligomer': 0},
                              'total': {'complex': 0, 'oligomer': 0}},
                     'type': None, 'hbond': 0, 'core': 0, 'interior': 0, 'rim': 0, 'support': 0}  # , 'hot_spot': 0}
    total_residue_dict = {}
    for entry in score_dict:
        residue_dict = {}
        # for column in columns:
        for key, value in score_dict[entry].items():
            # metadata = column.split('_')
            if key.startswith('per_res_'):
                metadata = key.split('_')
                res = int(metadata[-1])
                r_type = metadata[2]  # energy or sasa
                pose_state = metadata[-2]  # oligomer or complex
                if pose_state == 'oligomer' and offset:
                        res += offset[metadata[-3]]  # get oligomer chain offset
                if res not in residue_dict:
                    residue_dict[res] = copy.deepcopy(dict_template)
                if r_type == 'sasa':
                    # Ex. per_res_sasa_hydrophobic_A_oligomer_15 or per_res_sasa_hydrophobic_complex_15
                    polarity = metadata[3]
                    residue_dict[res][r_type][polarity][pose_state] = round(score_dict[entry][key], 3)
                    # residue_dict[res][r_type][polarity][pose_state] = round(score_dict[entry][column], 3)
                else:
                    # Ex. per_res_energy_A_oligomer_15 or per_res_energy_complex_15
                    residue_dict[res][r_type][pose_state] = round(score_dict[entry][key], 3)
        # if residue_dict:
        for res in residue_dict:
            try:
                residue_dict[res]['type'] = mutations[entry][res]
            except KeyError:
                residue_dict[res]['type'] = mutations['ref'][res]  # fill with aa from wild_type sequence
            if hbonds:
                if res in hbonds[entry]:
                    residue_dict[res]['hbond'] = 1
            residue_dict[res]['energy_delta'] = residue_dict[res]['energy']['complex'] \
                - residue_dict[res]['energy']['oligomer']  # - residue_dict[res]['energy']['fsp']
            rel_oligomer_sasa = calc_relative_sa(residue_dict[res]['type'],
                                                 residue_dict[res]['sasa']['total']['oligomer'])
            rel_complex_sasa = calc_relative_sa(residue_dict[res]['type'],
                                                residue_dict[res]['sasa']['total']['complex'])
            for polarity in residue_dict[res]['sasa']:
                # convert sasa measurements into bsa measurements
                residue_dict[res]['bsa_' + polarity] = round(residue_dict[res]['sasa'][polarity]['oligomer']
                                                             - residue_dict[res]['sasa'][polarity]['complex'], 2)
            if residue_dict[res]['bsa_total'] > 0:
                if rel_oligomer_sasa < 0.25:
                    residue_dict[res]['support'] = 1
                elif rel_complex_sasa < 0.25:
                    residue_dict[res]['core'] = 1
                else:
                    residue_dict[res]['rim'] = 1
            else:  # Todo remove res from dictionary as no interface design should be done? keep interior res constant?
                if rel_complex_sasa < 0.25:
                    residue_dict[res]['interior'] = 1
                # else:
                #     residue_dict[res]['surface'] = 1

            residue_dict[res].pop('sasa')
            residue_dict[res].pop('energy')
            # if residue_dict[res]['energy'] <= hot_spot_energy:
            #     residue_dict[res]['hot_spot'] = 1
        total_residue_dict[entry] = residue_dict

    return total_residue_dict


def mutation_conserved(residue_dict, background):  # TODO AMS
    """Process residue mutations compared to evolutionary background. Returns 1 if residue is observed in background

    Both residue_dict and background must be same index
    Args:
        residue_dict (dict): {15: {'type': 'T', ...}, ...}
        background (dict): {0: {'A': 0, 'R': 0, ...}, ...}
    Returns:
        conservation_dict (dict): {15: 1, 21: 0, 25: 1, ...}
    """
    conservation_dict = {}
    for residue in residue_dict:
        if residue in background:
            if background[residue][residue_dict[residue]['type']] > 0:
                conservation_dict[residue] = 1
                continue
        conservation_dict[residue] = 0

    return conservation_dict


def per_res_metric(sequence_dict, key=None):  # TODO AMS
    """Find metric value/residue in sequence dictionary specified by key

    Args:
        sequence_dict (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    Keyword Args:
        key='jsd' (str): Name of the residue metric to average
    Returns:
        jsd_per_res (float): 0.367
    """
    s, total = 0.0, 0
    if key:
        for residue in sequence_dict:
            if key in sequence_dict[residue]:
                s += sequence_dict[residue][key]
                total += 1
    else:
        for total, residue in enumerate(sequence_dict, 1):
            s += sequence_dict[residue]

    if total == 0:
        return 0.0
    else:
        return round(s / total, 3)


def df_permutation_test(grouped_df, diff_s, group1_size=0, compare='mean', permutations=1000):  # TODO SDUtils
    """From a two group dataframe, run a permutation test with mean comparison as default

    Args:
        grouped_df (pandas.DataFrame): DataFrame with only two groups contained. Doesn't need to be sorted
        diff_s (pandas.Series): Series with difference between two groups 'compare' stat for each column in grouped_df
    Keyword Args:
        group1_size=0 (int): Size of the observations in group1
        compare='mean' (str): Choose from any pandas.DataFrame attribute that collapses along a column
            Other options might be median.
    Returns:
        p_value_s (Series): Contains the p-value(s) of the permutation test using the 'compare' statistic against diff_s
    """
    permut_s_array = []
    for i in range(permutations):
        shuffled_df = grouped_df.sample(n=len(grouped_df))
        permut_s_array.append(getattr(shuffled_df.iloc[:group1_size, :], compare)().sub(
            getattr(shuffled_df.iloc[group1_size:, :], compare)()))
    # How many times the absolute permuted set is less than the absolute difference set. Returns bool, which when taking
    # the mean of, True value reflects 1 while False (more than/equal to the difference set) is 0.
    # In essence, the returned mean is the p-value, which indicates how significant the permutation test results are
    abs_s = diff_s.abs()
    bool_df = pd.DataFrame([permut_s.abs().lt(abs_s) for permut_s in permut_s_array])

    return bool_df.mean()


def hydrophobic_collapse_index(sequence):  # UNUSED TODO Validate, AMS
    """Calculate hydrophobic collapse index for a particular sequence of an iterable object and return a HCI array

    """
    seq_length = len(sequence)
    lower_range = 3
    upper_range = 9
    range_correction = 1
    range_size = upper_range - lower_range + range_correction
    yes = 1
    no = 0
    hydrophobic = ['F', 'I', 'L', 'V']
    sequence_array = []  # 0]  # initializes the list with index 0 equal to zero for sequence calculations
    # make an array with # of rows equal to upper range (+1 for indexing), length equal to # of letters in sequence
    window_array = np.zeros((upper_range + 1, seq_length))  # [[0] * (seq_length + 1) for i in range(upper_range + 1)]
    hci = np.zeros(seq_length)  # [0] * (seq_length + 1)
    for aa in sequence:
        if aa in hydrophobic:
            sequence_array.append(yes)
        else:
            sequence_array.append(no)

    for j in range(lower_range, upper_range + range_correction):
        # iterate over the window range
        window_len = math.floor(j / 2)
        modulus = j % 2
        # check if the range is odd or even, then calculate score accordingly, with cases for N- and C-terminal windows
        if modulus == 1:
            for k in range(seq_length):
                s = 0
                if k < window_len:
                    for n in range(k + window_len + range_correction):
                        s += sequence_array[n]
                elif k + window_len >= seq_length:
                    for n in range(k - window_len, seq_length):
                        s += sequence_array[n]
                else:
                    for n in range(k - window_len, k + window_len + range_correction):
                        s += sequence_array[n]
                window_array[j][k] = s / j
        else:
            for k in range(seq_length):
                s = 0
                if k < window_len:
                    for n in range(k + window_len + range_correction):
                        if n == k + window_len:
                            s += 0.5 * sequence_array[n]
                        else:
                            s += sequence_array[n]
                elif k + window_len >= seq_length:
                    for n in range(k - window_len, seq_length):
                        if n == k - window_len:
                            s += 0.5 * sequence_array[n]
                        else:
                            s += sequence_array[n]
                else:
                    for n in range(k - window_len, k + window_len + range_correction):
                        if n == k - window_len or n == k + window_len + range_correction:
                            s += 0.5 * sequence_array[n]
                        else:
                            s += sequence_array[n]
                window_array[j][k] = s / j

    # find the total for each position
    for k in range(seq_length):
        for j in range(lower_range, upper_range + range_correction):
            hci[k] += window_array[j][k]
        hci[k] /= range_size
        hci[k] = round(hci[k], 3)

    return hci


def calculate_column_number(num_groups=1, misc=0, sig=0):  # UNUSED, DEPRECIATED
    total = len(final_metrics) * len(stats_metrics)
    total += len(protocol_specific_columns) * num_groups * len(stats_metrics)
    total += misc
    total += sig  # for protocol pair mean similarity measure
    total += sig * len(protocol_specific_columns)  # for protocol pair individual similarity measure

    return total


# @handle_errors(error_type=(SDUtils.DesignError, AssertionError))
# TODO multiprocessing compliant (picklable) error decorator
@SDUtils.handle_errors(errors=(SDUtils.DesignError, AssertionError))
def analyze_output_s(des_dir, delta_refine=False, merge_residue_data=False, debug=False, save_trajectories=True,
                     figures=True):
    return analyze_output(des_dir, delta_refine=delta_refine, merge_residue_data=merge_residue_data, debug=debug,
                          save_trajectories=save_trajectories, figures=figures)


def analyze_output_mp(des_dir, delta_refine=False, merge_residue_data=False, debug=False, save_trajectories=True,
                      figures=True):
    try:
        pose = analyze_output(des_dir, delta_refine=delta_refine, merge_residue_data=merge_residue_data, debug=debug,
                              save_trajectories=save_trajectories, figures=figures)
        return pose, None
    except (SDUtils.DesignError, AssertionError) as e:
        return None, (des_dir.path, e)
    # finally:
    #     print('Error occurred in %s' % des_dir.path)
    #     return None, (des_dir.path, e)


def analyze_output(des_dir, delta_refine=False, merge_residue_data=False, debug=False, save_trajectories=True,
                   figures=True):
    """Retrieve all score information from a design directory and write results to .csv file

    Args:
        des_dir (DesignDirectory): DesignDirectory object
    Keyword Args:
        delta_refine (bbol): Whether to compute DeltaG for residues
        merge_residue_data (bool): Whether to incorporate residue data into Pose dataframe
        debug=False (bool): Whether to debug output
        save_trajectories=False (bool): Whether to save trajectory and residue dataframes
        figures=True (bool): Whether to make and save pose figures
    Returns:
        scores_df (Dataframe): Dataframe containing the average values from the input design directory
    """
    # Log output
    if debug:
        # global logger
        logger = SDUtils.start_log(name=__name__, handler=2, level=1,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    else:
        logger = SDUtils.start_log(name=__name__, handler=2, level=2,
                                   location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))

    # TODO add fraction_buried_atoms
    # Set up pose, ensure proper input
    global columns_to_remove, columns_to_rename, protocols_of_interest
    remove_columns = copy.deepcopy(columns_to_remove)
    rename_columns = copy.deepcopy(columns_to_rename)

    # Get design information including: interface residues, SSM's, and wild_type/design files
    design_flags = SDUtils.parse_flags_file(des_dir.path, name='design')
    des_residues = SDUtils.get_interface_residues(design_flags)  # Changed in favor of residue_processing identification

    # # used to be found from strings, now associated with the des_dir
    # pssm = SDUtils.parse_pssm(des_dir.info['pssm'])
    # # if os.path.exists(os.path.join(des_dir.path, PUtils.msa_pssm)):
    # #     pssm = SDUtils.parse_pssm(os.path.join(des_dir.path, PUtils.msa_pssm))
    # # else:
    # #     pssm = SDUtils.parse_pssm(os.path.join(des_dir.building_blocks, PUtils.msa_pssm))

    # # frag_pickle = glob(os.path.join(des_dir.data, '*%s*' % PUtils.frag_type))
    # # assert len(frag_pickle) == 1, 'Couldn\'t match file *%s*' % PUtils.frag_type
    # # # assert len(frag_pickle) == 1, '%s: error matching file %s' % (des_dir.path, '*' + PUtils.frag_type + '*')
    # # frag_pickle = frag_pickle[0]
    # # issm = SDUtils.unpickle(frag_pickle)  # issm only has residue info if interface info was available for residue
    # issm = SDUtils.unpickle(des_dir.info['issm'])
    # issm_residues = list(set(issm.keys()))
    # assert len(issm_residues) > 0, 'issm has no fragment information'
    # # dssm = SDUtils.parse_pssm(os.path.join(des_dir.path, PUtils.dssm))
    # dssm = SDUtils.parse_pssm(des_dir.info['dssm'])

    # frag_db = os.path.basename(des_dir.info['issm'].split(PUtils.frag_type)[0])
    # interface_bkgd = SDUtils.get_db_aa_frequencies(PUtils.frag_directory[os.path.basename(des_dir.info['db'])])
    interface_bkgd = SequenceProfile.get_db_aa_frequencies(PUtils.frag_directory[des_dir.info['db']])
    # profile_dict = {'evolution': pssm, 'fragment': issm, 'combined': dssm}
    profile_dict = {'evolution': SequenceProfile.parse_pssm(des_dir.info['pssm']),
                    'fragment': SDUtils.unpickle(des_dir.info['issm']),
                    'combined': SequenceProfile.parse_pssm(des_dir.info['dssm'])}
    issm_residues = list(set(profile_dict['fragment'].keys()))
    assert len(issm_residues) > 0, 'issm has no fragment information'

    # Get the scores from all design trajectories
    all_design_scores = read_scores(os.path.join(des_dir.scores, PUtils.scores_file))
    all_design_scores = SDUtils.clean_interior_keys(all_design_scores, remove_score_columns)

    # Gather mutations for residue specific processing and design sequences
    wild_type_file = SequenceProfile.get_wildtype_file(des_dir)
    wt_sequence = SequenceProfile.get_pdb_sequences(wild_type_file)
    all_design_files = SDUtils.get_directory_pdb_file_paths(des_dir.design_pdbs)
    # logger.debug('Design Files: %s' % ', '.join(all_design_files))
    sequence_mutations = SequenceProfile.generate_mutations(all_design_files, wild_type_file)
    # logger.debug('Design Files: %s' % ', '.join(sequence_mutations))
    offset_dict = SequenceProfile.pdb_to_pose_num(sequence_mutations['ref'])
    logger.debug('Chain offset: %s' % str(offset_dict))

    # Remove wt sequence and find all designs which have corresponding pdb files
    sequence_mutations.pop('ref')
    all_design_sequences = SequenceProfile.generate_sequences(wt_sequence, sequence_mutations)  # TODO just pull from design pdbs...
    logger.debug('all_design_sequences: %s' % ', '.join(name for chain in all_design_sequences
                                                        for name in all_design_sequences[chain]))
    all_design_scores = remove_pdb_prefixes(all_design_scores)
    all_design_sequences = {chain: remove_pdb_prefixes(all_design_sequences[chain]) for chain in all_design_sequences}
    # for chain in all_design_sequences:
    #     all_design_sequences[chain] = remove_pdb_prefixes(all_design_sequences[chain])

    # logger.debug('all_design_sequences2: %s' % ', '.join(name for chain in all_design_sequences
    #                                                      for name in all_design_sequences[chain]))
    logger.debug('all_design_scores: %s' % ', '.join(design for design in all_design_scores))
    # Ensure data is present for both scores and sequences, then initialize DataFrames
    good_designs = list(set([design for chain in all_design_sequences for design in all_design_sequences[chain]])
                        & set([design for design in all_design_scores]))
    logger.info('All Designs: %s' % ', '.join(good_designs))
    all_design_scores = SDUtils.clean_dictionary(all_design_scores, good_designs, remove=False)
    all_design_sequences = {chain: SDUtils.clean_dictionary(all_design_sequences[chain], good_designs, remove=False)
                            for chain in all_design_sequences}
    logger.debug('All Sequences: %s' % all_design_sequences)
    # pd.set_option('display.max_columns', None)
    idx = pd.IndexSlice
    scores_df = pd.DataFrame(all_design_scores).T

    # Gather all columns into specific types for processing and formatting TODO move up
    report_columns, per_res_columns, hbonds_columns = {}, [], []
    for column in list(scores_df.columns):
        if column.startswith('R_'):
            report_columns[column] = column.replace('R_', '')
        elif column.startswith('symmetry_switch'):
            symmetry = scores_df.loc[PUtils.stage[1], column].replace('make_', '')
        elif column.startswith('per_res_'):
            per_res_columns.append(column)
        elif column.startswith('hbonds_res_selection'):
            hbonds_columns.append(column)
    rename_columns.update(report_columns)
    rename_columns.update({'R_int_sc': 'shape_complementarity', 'R_full_stability': 'full_stability_complex',
                           'R_full_stability_oligomer_A': 'full_stability_oligomer_A',
                           'R_full_stability_oligomer_B': 'full_stability_oligomer_B',
                           'R_full_stability_A_oligomer': 'full_stability_oligomer_A',
                           'R_full_stability_B_oligomer': 'full_stability_oligomer_B',
                           'R_int_energy_context_A_oligomer': 'int_energy_context_oligomer_A',
                           'R_int_energy_context_B_oligomer': 'int_energy_context_oligomer_B'})
    #                       TODO remove the update when metrics protocol is changed
    res_columns = hbonds_columns + per_res_columns
    remove_columns += res_columns + [groups]

    # Format columns
    scores_df = scores_df.rename(columns=rename_columns)
    scores_df = scores_df.groupby(level=0, axis=1).apply(lambda x: x.apply(join_columns, axis=1))
    # Check proper input
    metric_set = necessary_metrics.copy() - set(scores_df.columns)
    assert metric_set == set(), 'Missing required metrics: %s' % metric_set
    # assert metric_set == set(), logger.critical('%s: Missing required metrics: %s' % (des_dir.path, metric_set))
    # CLEAN: Create new columns, remove unneeded columns, create protocol dataframe
    # TODO protocol switch or no_design switch?
    protocol_s = scores_df[groups]
    logger.debug(protocol_s)
    designs = protocol_s.index.to_list()
    logger.debug('Design indices: %s' % designs)
    # Modify protocol name for refine and consensus
    for stage in PUtils.stage_f:
        if stage in designs:
            # change design index value to PUtils.stage[i] (for consensus and refine)
            protocol_s[stage] = stage  # TODO remove in future scripts
            # protocol_s.at[PUtils.stage[stage], groups] = PUtils.stage[stage]
    # Replace empty strings with numpy.notanumber (np.nan), drop all str columns, and convert remaining data to float
    scores_df = scores_df.replace('', np.nan)
    scores_df = scores_df.drop(remove_columns, axis=1, errors='ignore').astype(float)
    if delta_refine:
        scores_df = scores_df.sub(scores_df.loc[PUtils.stage[1], ])
    scores_df = columns_to_new_column(scores_df, summation_pairs)
    scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
    # Remove unnecessary and Rosetta score terms TODO learn know how to produce them. Not in FastRelax...
    scores_df.drop(unnecessary + rosetta_terms, axis=1, inplace=True, errors='ignore')

    # TODO remove dirty when columns are correct (after P432) and column tabulation precedes residue/hbond_processing
    interface_hbonds = dirty_hbond_processing(all_design_scores)  # , offset=offset_dict) when hbonds are pose numbering
    # interface_hbonds = hbond_processing(all_design_scores, hbonds_columns)  # , offset=offset_dict)

    all_mutations = SequenceProfile.generate_mutations(all_design_files, wild_type_file, pose_num=True)
    all_mutations_no_chains = SequenceProfile.make_mutations_chain_agnostic(all_mutations)
    all_mutations_simplified = SequenceProfile.simplify_mutation_dict(all_mutations_no_chains)
    cleaned_mutations = remove_pdb_prefixes(all_mutations_simplified)
    residue_dict = dirty_residue_processing(all_design_scores, cleaned_mutations, offset=offset_dict,
                                            hbonds=interface_hbonds)
    # residue_dict = residue_processing(all_design_scores, cleaned_mutations, per_res_columns, offset=offset_dict,
    #                                   hbonds=interface_hbonds)  # TODO when columns are correct

    # Calculate amino acid observation percent from residue dict and background SSM's
    obs_d = {}
    for profile in profile_dict:
        obs_d[profile] = {design: mutation_conserved(residue_dict[design], SequenceProfile.offset_index(profile_dict[profile]))
                          for design in residue_dict}

    # Remove residues from fragment dict if no fragment information available for them
    obs_d['fragment'] = SDUtils.clean_interior_keys(obs_d['fragment'], issm_residues, remove=False)
    # Add observation information into the residue dictionary
    for design in residue_dict:
        res_dict = {'observed_%s' % profile: obs_d[profile][design] for profile in obs_d}
        residue_dict[design] = SequenceProfile.weave_sequence_dict(base_dict=residue_dict[design], **res_dict)

    # Find the observed background for each design in the pose
    pose_observed_bkd = {profile: {design: per_res_metric(obs_d[profile][design]) for design in obs_d[profile]}
                         for profile in profile_dict}
    for profile in profile_dict:
        scores_df['observed_%s' % profile] = pd.Series(pose_observed_bkd[profile])

    # Process H-bond and Residue metrics to dataframe
    residue_df = pd.concat({key: pd.DataFrame(value) for key, value in residue_dict.items()}).unstack()
    # residue_df - returns multi-index column with residue number as first (top) column index, metric as second index
    # during residue_df unstack, all residues with missing dicts are copied as nan
    number_hbonds = {entry: len(interface_hbonds[entry]) for entry in interface_hbonds}
    # number_hbonds_df = pd.DataFrame(number_hbonds, index=['number_hbonds', ]).T
    number_hbonds_s = pd.Series(number_hbonds, name='number_hbonds')
    scores_df = pd.merge(scores_df, number_hbonds_s, left_index=True, right_index=True)

    # Add design residue information to scores_df such as core, rim, and support measures
    for r_class in residue_classificiation:
        scores_df[r_class] = residue_df.loc[:, idx[:, residue_df.columns.get_level_values(1) == r_class]].sum(axis=1)
    scores_df['int_composition_diff'] = scores_df.apply(residue_composition_diff, axis=1)

    interior_residue_df = residue_df.loc[:, idx[:, residue_df.columns.get_level_values(1) == 'interior']].droplevel(1, axis=1)
    # Check if any of the values in columns are 1. If so, return True for that column
    interior_residues = interior_residue_df.any().index[interior_residue_df.any()].to_list()
    int_residues = list(set(residue_df.columns.get_level_values(0).unique()) - set(interior_residues))
    if set(int_residues) != set(des_residues):
        logger.info('Residues %s are located in the interior' %
                    ', '.join(map(str, list(set(des_residues) - set(int_residues)))))
    scores_df['total_interface_residues'] = len(int_residues)

    # Gather miscellaneous pose specific metrics
    other_pose_metrics = des_dir.pose_metrics()
    # other_pose_metrics = Pose.gather_fragment_metrics(des_dir)
    # nanohedra_score, average_fragment_z_score, unique_fragments
    other_pose_metrics['observations'] = len(designs)
    other_pose_metrics['symmetry'] = symmetry
    # other_pose_metrics['total_interface_residues'] = len(int_residues)
    other_pose_metrics['percent_fragment'] = len(profile_dict['fragment']) / len(int_residues)

    # Interface B Factor TODO ensure clean_asu.pdb has B-factors
    wt_pdb = SDUtils.read_pdb(wild_type_file)
    chain_sep = wt_pdb.getTermCAAtom('C', wt_pdb.chain_id_list[0]).residue_number  # this only works with 2 chains TODO
    int_b_factor = 0
    for residue in int_residues:
        if residue <= chain_sep:
            int_b_factor += wt_pdb.get_ave_residue_b_factor(wt_pdb.chain_id_list[0], residue)
        else:
            int_b_factor += wt_pdb.get_ave_residue_b_factor(wt_pdb.chain_id_list[1], residue)
    other_pose_metrics['interface_b_factor_per_res'] = round(int_b_factor / len(int_residues), 2)

    pose_alignment = SequenceProfile.multi_chain_alignment(all_design_sequences)
    mutation_frequencies = SDUtils.clean_dictionary(pose_alignment['counts'], int_residues, remove=False)
    # Calculate Jensen Shannon Divergence using different SSM occurrence data and design mutations
    pose_res_dict = {}
    for profile in profile_dict:  # both mut_freq and profile_dict[profile] are zero indexed
        pose_res_dict['divergence_%s' % profile] = SequenceProfile.pos_specific_jsd(mutation_frequencies, profile_dict[profile])

    pose_res_dict['divergence_interface'] = SDUtils.compute_jsd(mutation_frequencies, interface_bkgd)
    # pose_res_dict['hydrophobic_collapse_index'] = hci()  # TODO HCI

    # Subtract residue info from reference (refine)
    if delta_refine:
        # TODO Refine is not great ref for deltaG as modelling occurred. Only subtracting energy [res_columns_subtract]
        residue_df.update(residue_df.iloc[:, residue_df.columns.get_level_values(1) == 'energy'].sub(
            residue_df.loc[PUtils.stage[1], residue_df.columns.get_level_values(1) == 'energy']))

    # Divide/Multiply column pairs to new columns
    scores_df = columns_to_new_column(scores_df, division_pairs, mode='truediv')

    # Merge processed dataframes
    scores_df = pd.merge(protocol_s, scores_df, left_index=True, right_index=True)
    protocol_df = pd.DataFrame(protocol_s)
    protocol_df.columns = pd.MultiIndex.from_product([[''], protocol_df.columns])
    residue_df = pd.merge(protocol_df, residue_df, left_index=True, right_index=True)

    # Drop refine row and any rows with nan values
    scores_df.drop(PUtils.stage[1], axis=0, inplace=True, errors='ignore')
    residue_df.drop(PUtils.stage[1], axis=0, inplace=True, errors='ignore')
    clean_scores_df = scores_df.dropna()
    residue_df = residue_df.dropna(how='all', axis=1)  # remove completely empty columns (obs_interface)
    clean_residue_df = residue_df.dropna()
    # print(residue_df.isna())  #.any(axis=1).to_list())  # scores_df.where()
    scores_na_index = scores_df[~scores_df.index.isin(clean_scores_df.index)].index.to_list()
    residue_na_index = residue_df[~residue_df.index.isin(clean_residue_df.index)].index.to_list()
    if scores_na_index:
        protocol_s.drop(scores_na_index, inplace=True)
        logger.warning('%s: Trajectory DataFrame dropped rows with missing values: %s' %
                       (des_dir.path, ', '.join(scores_na_index)))
    if residue_na_index:
        logger.warning('%s: Residue DataFrame dropped rows with missing values: %s' %
                       (des_dir.path, ', '.join(residue_na_index)))

    # Fix reported per_residue_energy to contain only interface. BUT With delta, these residues should be subtracted
    # int_residue_df = residue_df.loc[:, idx[int_residues, :]]

    # Get unique protocols for protocol specific metrics and drop unneeded protocol values
    unique_protocols = protocol_s.unique().tolist()
    protocol_intersection = set(protocols_of_interest) & set(unique_protocols)
    # if len(unique_protocols) == 1:  # TODO protocol switch or no design switch
    assert protocol_intersection == set(protocols_of_interest), \
        'Missing %s protocol required for significance measurements! Analysis failed' \
        % ', '.join(set(protocols_of_interest) - protocol_intersection)
    for value in ['refine', '']:  # TODO remove '' after P432 MinMatch6 upon future script deployment
        try:
            unique_protocols.remove(value)
        except ValueError:
            pass
    logger.info('Unique Protocols: %s' % ', '.join(unique_protocols))

    designs_by_protocol, sequences_by_protocol = {}, {}
    stats_by_protocol = {protocol: {} for protocol in unique_protocols}
    for protocol in unique_protocols:
        designs_by_protocol[protocol] = protocol_s.index[protocol_s == protocol].tolist()
        sequences_by_protocol[protocol] = {chain: {name: all_design_sequences[chain][name]
                                                   for name in all_design_sequences[chain]
                                                   if name in designs_by_protocol[protocol]}
                                           for chain in all_design_sequences}
        protocol_alignment = SequenceProfile.multi_chain_alignment(sequences_by_protocol[protocol])
        protocol_mutation_freq = SequenceProfile.remove_non_mutations(protocol_alignment['counts'], int_residues)
        protocol_res_dict = {'divergence_%s' % profile: SequenceProfile.pos_specific_jsd(protocol_mutation_freq, profile_dict[profile])
                             for profile in profile_dict}  # both prot_freq and profile_dict[profile] are zero indexed
        protocol_res_dict['divergence_interface'] = SDUtils.compute_jsd(protocol_mutation_freq, interface_bkgd)

        # Get per residue divergence metric by protocol
        for key in protocol_res_dict:
            stats_by_protocol[protocol]['%s_per_res' % key] = per_res_metric(protocol_res_dict[key])  # , key=key)
            # {protocol: 'jsd_per_res': 0.747, 'int_jsd_per_res': 0.412}, ...}
        # Get per design observed background metric by protocol
        for profile in profile_dict:
            stats_by_protocol[protocol]['observed_%s' % profile] = per_res_metric(
                {des: pose_observed_bkd[profile][des] for des in designs_by_protocol[protocol]})

        # Gather the average number of residue classifications for each protocol
        for res_class in residue_classificiation:
            stats_by_protocol[protocol][res_class] = clean_residue_df.loc[
                designs_by_protocol[protocol],
                idx[:, clean_residue_df.columns.get_level_values(1) == res_class]].mean().sum()
        stats_by_protocol[protocol]['observations'] = len(designs_by_protocol[protocol])
    protocols_by_design = {v: k for k, _list in designs_by_protocol.items() for v in _list}

    # POSE ANALYSIS: Get total pose design statistics
    # remove below if consensus is run multiple times. the cst_weights are very large and destroy the mean
    trajectory_df = clean_scores_df.drop(PUtils.stage[5], axis=0, errors='ignore')
    assert len(trajectory_df.index.to_list()) > 0, 'No design was done on this pose'
    # assert len(trajectory_df.index.to_list()) > 0, '%s: No design was done on this pose' % des_dir.path
    # TODO protocol switch or no design switch
    traj_stats = {}
    protocol_stat_df = {}
    for stat in stats_metrics:
        traj_stats[stat] = getattr(trajectory_df, stat)().rename(stat)
        protocol_stat_df[stat] = getattr(clean_scores_df.groupby(groups), stat)()
        if stat == 'mean':
            continue
        protocol_stat_df[stat].index = protocol_stat_df[stat].index.to_series().map(
            {protocol: protocol + '_' + stat for protocol in sorted(unique_protocols)})
    trajectory_df = trajectory_df.append([traj_stats[stat] for stat in traj_stats])
    # Here we add consensus back to the trajectory_df after removing above (line 1073)
    trajectory_df = trajectory_df.append([protocol_stat_df[stat] for stat in protocol_stat_df])

    if merge_residue_data:
        trajectory_df = pd.merge(trajectory_df, clean_residue_df, left_index=True, right_index=True)

    # Calculate protocol significance
    # Find all unique combinations of protocols using 'mean' as all protocol combination source. Excludes Consensus
    protocol_subset_df = trajectory_df.loc[:, protocol_specific_columns]
    sig_df = protocol_stat_df[stats_metrics[0]]
    assert len(sig_df.index.to_list()) > 1, 'Can\'t measure protocol significance'
    pvalue_df = pd.DataFrame()
    for pair in combinations(sorted(sig_df.index.to_list()), 2):
        select_df = protocol_subset_df.loc[designs_by_protocol[pair[0]] + designs_by_protocol[pair[1]], :]
        difference_s = sig_df.loc[pair[0], protocol_specific_columns].sub(
            sig_df.loc[pair[1], protocol_specific_columns])
        pvalue_df[pair] = df_permutation_test(select_df, difference_s, group1_size=len(designs_by_protocol[pair[0]]),
                                              compare=stats_metrics[0])
    logger.debug(pvalue_df)
    pvalue_df = pvalue_df.T  # change the significance pairs to the indices and protocol specific columns to columns
    trajectory_df = trajectory_df.append(pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1))

    # Get pose sequence divergence TODO protocol switch
    sim_sum_and_divergence_stats = {'%s_per_res' % key: per_res_metric(pose_res_dict[key]) for key in pose_res_dict}

    # Compute sequence differences between each protocol
    residue_energy_df = clean_residue_df.loc[:, idx[:, clean_residue_df.columns.get_level_values(1) == 'energy_delta']]
    # num_components = 3  # TODO choose number of componenents or percent variance explained
    # pca = PCA(num_components)
    res_pca = PCA(PUtils.variance)  # P432 designs used 0.8 percent of the variance
    residue_energy_np = StandardScaler().fit_transform(residue_energy_df.values)
    residue_energy_pc = res_pca.fit_transform(residue_energy_np)
    residue_energy_pc_df = pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
                                        columns=['pc' + str(x + SDUtils.index_offset)
                                                 for x in range(len(res_pca.components_))])
    #                                    ,columns=residue_energy_df.columns)

    seq_pca = copy.deepcopy(res_pca)
    residue_dict.pop(PUtils.stage[1])  # Remove refine from analysis before PC calculation
    pairwise_sequence_diff_np = SDUtils.all_vs_all(residue_dict, SequenceProfile.sequence_difference)
    pairwise_sequence_diff_np = StandardScaler().fit_transform(pairwise_sequence_diff_np)
    seq_pc = seq_pca.fit_transform(pairwise_sequence_diff_np)
    # Compute the euclidean distance
    # pairwise_pca_distance_np = pdist(seq_pc)
    # pairwise_pca_distance_np = SDUtils.all_vs_all(seq_pc, euclidean)

    # Make PC DataFrame
    # First take all the principal components identified from above and merge with labels
    # Next the labels will be grouped and stats are taken for each group (mean is important)
    # All protocol means will have pairwise distance measured as a means of accessing similarity
    # These distance metrics will be reported in the final pose statistics
    seq_pc_df = pd.DataFrame(seq_pc, index=list(residue_dict.keys()),
                             columns=['pc' + str(x + SDUtils.index_offset) for x in range(len(seq_pca.components_))])
    # Merge principle components with labels
    residue_energy_pc_df = pd.merge(protocol_s, residue_energy_pc_df, left_index=True, right_index=True)
    seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

    # Gather protocol similarity/distance metrics
    sim_measures = {'similarity': None, 'seq_distance': {}, 'energy_distance': {}}
    # Find similarity between each type of protocol by taking row average of all p-values for each metric
    mean_pvalue_s = pvalue_df.mean(axis=1)  # protocol pair : mean significance
    mean_pvalue_s.index = pd.MultiIndex.from_tuples(mean_pvalue_s.index)
    sim_measures['similarity'] = mean_pvalue_s
    # sim_measures['similarity'] = pvalue_df.mean(axis=1)

    # TODO protocol switch or no design switch
    grouped_pc_stat_df_dict, grouped_pc_energy_df_dict = {}, {}
    for stat in stats_metrics:
        grouped_pc_stat_df_dict[stat] = getattr(seq_pc_df.groupby(groups), stat)()
        grouped_pc_energy_df_dict[stat] = getattr(residue_energy_pc_df.groupby(groups), stat)()
        if stat == 'mean':
            # if renaming is necessary
            # protocol_stat_df[stat].index = protocol_stat_df[stat].index.to_series().map(
            #     {protocol: protocol + '_' + stat for protocol in sorted(unique_protocols)})
            seq_pca_mean_distance_vector = pdist(grouped_pc_stat_df_dict[stat])
            energy_pca_mean_distance_vector = pdist(grouped_pc_energy_df_dict[stat])
            # protocol_indices_map = list(tuple(condensed_to_square(k, len(seq_pca_mean_distance_vector)))
            #                             for k in seq_pca_mean_distance_vector)
            for k, dist in enumerate(seq_pca_mean_distance_vector):
                i, j = SDUtils.condensed_to_square(k, len(grouped_pc_stat_df_dict[stat].index))
                sim_measures['seq_distance'][(grouped_pc_stat_df_dict[stat].index[i],
                                              grouped_pc_stat_df_dict[stat].index[j])] = dist

            for k, e_dist in enumerate(energy_pca_mean_distance_vector):
                i, j = SDUtils.condensed_to_square(k, len(grouped_pc_energy_df_dict[stat].index))
                sim_measures['energy_distance'][(grouped_pc_energy_df_dict[stat].index[i],
                                                 grouped_pc_energy_df_dict[stat].index[j])] = e_dist

    for pc_stat in grouped_pc_stat_df_dict:
        logger.info(grouped_pc_stat_df_dict[pc_stat])

    # Find total protocol similarity for different metrics
    for measure in sim_measures:
        measure_s = pd.Series({pair: sim_measures[measure][pair] for pair in combinations(protocols_of_interest, 2)})
        sim_sum_and_divergence_stats['protocol_%s_sum' % measure] = measure_s.sum()

    # Create figures
    if figures:
        _path = os.path.join(des_dir.all_scores, str(des_dir))
        # Set up Labels & Plot the PC data
        protocol_map = {protocol: i for i, protocol in enumerate(unique_protocols)}
        integer_map = {i: protocol for (protocol, i) in protocol_map.items()}
        pc_labels_group = [protocols_by_design[design] for design in residue_dict]
        # pc_labels_group = np.array([protocols_by_design[design] for design in residue_dict])
        pc_labels_int = [protocol_map[protocols_by_design[design]] for design in residue_dict]
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
        # plt.cla()

        # for color_int, label in integer_map.items():  # zip(pc_labels_group, pc_labels_int):
        #     ax.scatter(seq_pc[pc_labels_group == label, 0],
        #                seq_pc[pc_labels_group == label, 1],
        #                seq_pc[pc_labels_group == label, 2],
        #                c=color_int, cmap=plt.cm.nipy_spectral, edgecolor='k')
        scatter = ax.scatter(seq_pc[:, 0], seq_pc[:, 1], seq_pc[:, 2], c=pc_labels_int, cmap='Spectral', edgecolor='k')
        # handles, labels = scatter.legend_elements()
        # # print(labels)  # ['$\\mathdefault{0}$', '$\\mathdefault{1}$', '$\\mathdefault{2}$']
        # ax.legend(handles, labels, loc='upper right', title=groups)
        # # ax.legend(handles, [integer_map[label] for label in labels], loc="upper right", title=groups)
        # # plt.axis('equal') # not possible with 3D graphs
        # plt.legend()  # No handles with labels found to put in legend.
        colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
        custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in colors]
        ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left', bbox_to_anchor=(1.0, .5))
        # # Add group mean to the plot
        # for name, label in integer_map.items():
        #     ax.scatter(seq_pc[pc_labels_group == label, 0].mean(), seq_pc[pc_labels_group == label, 1].mean(),
        #                seq_pc[pc_labels_group == label, 2].mean(), marker='x')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        # plt.legend(pc_labels_group)
        plt.savefig('%s_seq_pca.png' % _path)
        plt.clf()
        # Residue PCA Figure to assay multiple interface states
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = Axes3D(fig, rect=[0, 0, .7, 1], elev=48, azim=134)
        scatter = ax.scatter(residue_energy_pc[:, 0], residue_energy_pc[:, 1], residue_energy_pc[:, 2], c=pc_labels_int,
                             cmap='Spectral', edgecolor='k')
        colors = [scatter.cmap(scatter.norm(i)) for i in integer_map.keys()]
        custom_lines = [plt.Line2D([], [], ls='', marker='.', mec='k', mfc=c, mew=.1, ms=20) for c in colors]
        ax.legend(custom_lines, [j for j in integer_map.values()], loc='center left', bbox_to_anchor=(1.0, .5))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.savefig('%s_res_energy_pca.png' % _path)

    # Save Trajectory, Residue DataFrames, and PDB Sequences
    if save_trajectories:
        # trajectory_df.to_csv('%s_Trajectories.csv' % _path)
        trajectory_df.to_csv(des_dir.trajectories)
        # clean_residue_df.to_csv('%s_Residues.csv' % _path)
        clean_residue_df.to_csv(des_dir.residues)
        SDUtils.pickle_object(all_design_sequences, '%s_Sequences' % str(des_dir), out_path=des_dir.all_scores)

    # CONSTRUCT: Create pose series and format index names
    pose_stat_s, protocol_stat_s = {}, {}
    for stat in stats_metrics:
        pose_stat_s[stat] = trajectory_df.loc[stat, :]
        pose_stat_s[stat] = pd.concat([pose_stat_s[stat]], keys=['pose'])
        pose_stat_s[stat] = pd.concat([pose_stat_s[stat]], keys=[stat])
        # Collect protocol specific metrics in series
        suffix = ''
        if stat != 'mean':
            suffix = '_' + stat
        protocol_stat_s[stat] = pd.concat([protocol_subset_df.loc[protocol + suffix, :]
                                           for protocol in unique_protocols], keys=unique_protocols)
        protocol_stat_s[stat] = pd.concat([protocol_stat_s[stat]], keys=[stat])

    # Find the significance between each pair of protocols
    protocol_sig_s = pd.concat([pvalue_df.loc[[pair], :].squeeze() for pair in pvalue_df.index.to_list()],
                               keys=[tuple(pair) for pair in pvalue_df.index.to_list()])
    # squeeze turns the column headers into series indices. Keys appends to make a multi-index
    protocol_stats_s = pd.concat([pd.Series(stats_by_protocol[protocol]) for protocol in stats_by_protocol],
                                 keys=unique_protocols)
    other_metrics_s = pd.Series(other_pose_metrics)
    other_stats_s = pd.Series(sim_sum_and_divergence_stats)

    # Add series specific Multi-index names to data
    protocol_stats_s = pd.concat([protocol_stats_s], keys=['stats'])
    other_metrics_s = pd.concat([other_metrics_s], keys=['pose'])
    other_metrics_s = pd.concat([other_metrics_s], keys=['dock'])
    other_stats_s = pd.concat([other_stats_s], keys=['pose'])
    other_stats_s = pd.concat([other_stats_s], keys=['seq_design'])

    # Process similarity between protocols
    sim_measures_s = pd.concat([pd.Series(sim_measures[measure]) for measure in sim_measures],
                               keys=[measure for measure in sim_measures])

    # Combine all series
    pose_s = pd.concat([pose_stat_s[stat] for stat in pose_stat_s] + [protocol_stat_s[stat] for stat in protocol_stat_s]
                       + [protocol_sig_s, protocol_stats_s, other_metrics_s, other_stats_s, sim_measures_s]).swaplevel(0, 1)

    # Remove pose specific metrics from pose_s, sort, and name protocol_mean_df TODO protocol switch or no design switch
    pose_s.drop([groups, ], level=2, inplace=True)
    pose_s.sort_index(level=2, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
    pose_s.sort_index(level=1, inplace=True, sort_remaining=False)  # ascending=True, sort_remaining=True)
    pose_s.sort_index(level=0, inplace=True, sort_remaining=False)  # ascending=False
    pose_s.name = str(des_dir)

    # misc_columns = len(stats_by_protocol[unique_protocols[-1]]) * len(unique_protocols) + len(other_pose_metrics)  # \
    # + len(pvalue_df.columns.to_list())  # last term is 'similarity' of protocol pairs. was pvalue_df.index.to_list()
    # intended_columns = calculate_column_number(num_groups=len(unique_protocols), misc=misc_columns,
    #                                          sig=len(pvalue_df.index.to_list()))  # sig=len(pvalue_df.columns.to_list(
    # if pose_s.size != intended_columns:  # TODO add distance columns
    #     logger.error('%s: The number of columns %d does not match the intended amount (%d), underlying data might be '
    #                  'mutated. Proceed with CAUTION!' % (des_dir.path, pose_s.size, intended_columns))

    return pose_s


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
    parser.add_argument('-n', '--no_save', action='store_true', help='Don\'t save trajectory information. '
                                                                     'Default=False')
    parser.add_argument('-b', '--debug', action='store_true', help='Debug all steps to standard out? Default=False')
    parser.add_argument('-j', '--join', action='store_true', help='Join Trajectory and Residue Dataframes? '
                                                                  'Default=False')
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
                (os.path.basename(__file__),
                 '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    # Collect all designs to be processed
    all_poses, location = SDUtils.collect_designs(args.directory, file=args.file)
    assert all_poses != list(), logger.critical('No %s directories found within \'%s\' input! Please ensure correct '
                                                'location.' % (PUtils.nano, location))
    logger.info('%d Poses found in \'%s\'' % (len(all_poses), location))
    logger.info('All pose specific logs are located in their corresponding directories.\nEx: \'%s\'' %
                os.path.join(all_poses[0].path, os.path.basename(all_poses[0].path) + '.log'))
    all_design_directories = DesignDirectory.set_up_directory_objects(all_poses)

    if args.no_save:
        save = False
    else:
        save = True
    # Start pose analysis of all designed files
    if args.multi_processing:
        # Calculate the number of threads to use depending on computer resources
        mp_threads = SDUtils.calculate_mp_threads(maximum=True)
        logger.info('Starting multiprocessing using %s threads' % str(mp_threads))
        zipped_args = zip(all_design_directories, repeat(args.delta_g), repeat(args.join), repeat(args.debug),
                          repeat(save))
        pose_results, exceptions = SDUtils.mp_try_starmap(analyze_output_mp, zipped_args, mp_threads)
    else:
        logger.info('Starting processing. If single process is taking awhile, use -m during submission')
        pose_results, exceptions = [], []
        for des_directory in all_design_directories:
            result, error = analyze_output_s(des_directory, delta_refine=args.delta_g, merge_residue_data=args.join,
                                             debug=args.debug, save_trajectories=save)
            pose_results.append(result)
            exceptions.append(error)

    failures = [i for i, exception in enumerate(exceptions) if exception]
    for index in reversed(failures):
        del pose_results[index]
        # pose_results.remove(index)

    exceptions = list(set(exceptions))
    if len(exceptions) == 1 and exceptions[0]:
        logger.warning('\nThe following exceptions were thrown. Design for these directories is inaccurate\n')
        for exception in exceptions:
            logger.warning(exception)

    if len(all_poses) >= 1:
        design_df = pd.DataFrame(pose_results)
        # Save Design dataframe
        out_path = os.path.join('/gscratch/kmeador/crystal_design/NanohedraEntry65MinMatched6_2nd', PUtils.analysis_file)  # args.directory
        if os.path.exists(out_path):
            design_df.to_csv(out_path, mode='a', header=False)
        else:
            design_df.to_csv(out_path)
        logger.info('All Design Pose Analysis written to %s' % out_path)
