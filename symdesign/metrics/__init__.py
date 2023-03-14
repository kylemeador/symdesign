from __future__ import annotations

import logging
import math
import operator
import warnings
from itertools import repeat
from json import loads
from typing import AnyStr, Any, Sequence, Iterable, Mapping, Literal

from numba import jit
import numpy as np
import pandas as pd
import torch

from . import pose, sql
from symdesign.resources import config
from symdesign.resources.query.utils import input_string, validate_type, verify_choice, header_string
from symdesign.structure.utils import DesignError
from symdesign import flags, sequence, utils
putils = utils.path

logger = logging.getLogger(__name__)
residue_classification = ['core', 'rim', 'support']  # 'hot_spot'
per_residue_energy_states = \
    ['complex', 'bound', 'unbound', 'energy_delta', 'solv_complex', 'solv_bound', 'solv_unbound']
energy_metric_names = ['interface_energy_complex', 'interface_energy_bound', 'interface_energy_unbound',
                       'interface_energy',
                       'interface_solvation_energy_complex', 'interface_solvation_energy_bound',
                       'interface_solvation_energy_unbound']
relative_sasa_states = ['sasa_relative_complex', 'sasa_relative_bound']
per_residue_sasa_states = ['sasa_hydrophobic_bound', 'sasa_polar_bound', 'sasa_total_bound',
                           'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex']
sasa_metric_names = ['area_hydrophobic_unbound', 'area_polar_unbound', 'area_total_unbound',
                     'area_hydrophobic_complex', 'area_polar_complex', 'area_total_complex']
per_residue_interface_states = ['bsa_polar', 'bsa_hydrophobic', 'bsa_total']
interface_sasa_metric_names = ['interface_area_polar', 'interface_area_hydrophobic', 'interface_area_total']
collapse_metrics = ['collapse_new_positions', 'collapse_new_position_significance',
                    'collapse_significance_by_contact_order_z', 'collapse_increase_significance_by_contact_order_z',
                    'collapse_increased_z', 'collapse_deviation_magnitude', 'collapse_sequential_peaks_z',
                    'collapse_sequential_z']
zero_probability_frag_value = -20
proteinmpnn_scores = ['sequences', 'proteinmpnn_loss_complex', 'proteinmpnn_loss_unbound', 'design_indices']
# Only slice the final 3 values
sasa_metrics_rename_mapping = dict([*zip(per_residue_interface_states, interface_sasa_metric_names),
                                    *zip(per_residue_sasa_states, sasa_metric_names)])
# Based on bsa_total values for highest deviating surface residue of one design from multiple measurements
# Ex: 0.45, 0.22, 0.04, 0.19, 0.01, 0.2, 0.04, 0.19, 0.01, 0.19, 0.01, 0.21, 0.06, 0.17, 0.01, 0.21, -0.04, 0.22
bsa_tolerance = 0.25
energy_metrics_rename_mapping = dict(zip(per_residue_energy_states, energy_metric_names))
# other_metrics_rename_mapping = dict(hbond='number_hbonds', design_residue='total_design_residues')
renamed_design_metrics = {
    'design_residue': 'number_residues_design', 'interface_residue': 'number_residues_interface',
    'hbond': 'number_hbonds', 'mutation': 'number_mutations', 'type': 'sequence'
}

errat_1_sigma, errat_2_sigma, errat_3_sigma = 5.76, 11.52, 17.28  # These are approximate magnitude of deviation
collapse_thresholds = {
    'standard': 0.43,
    'expanded': 0.48
}
collapse_reported_std = .05
idx_slice = pd.IndexSlice
filter_df = pd.DataFrame(config.metrics)
dock_metrics = {
    'proteinmpnn_dock_cross_entropy_loss',
    'proteinmpnn_dock_cross_entropy_per_residue',
    'proteinmpnn_v_design_probability_cross_entropy_loss',
    'proteinmpnn_v_design_probability_cross_entropy_per_residue',
    'proteinmpnn_v_evolution_probability_cross_entropy_loss',
    'proteinmpnn_v_evolution_probability_cross_entropy_per_residue',
    'proteinmpnn_v_fragment_probability_cross_entropy_loss',
    'proteinmpnn_v_fragment_probability_cross_entropy_per_residue',
    'dock_collapse_significance_by_contact_order_z_mean',
    'dock_collapse_increased_z_mean',
    'dock_collapse_sequential_peaks_z_mean',
    'dock_collapse_sequential_z_mean',
    'dock_collapse_deviation_magnitude',
    'dock_collapse_increase_significance_by_contact_order_z',
    'dock_collapse_increased_z',
    'dock_collapse_new_positions',
    'dock_collapse_new_position_significance',
    'dock_collapse_sequential_peaks_z',
    'dock_collapse_sequential_z',
    'dock_collapse_significance_by_contact_order_z',
    'dock_collapse_variance',
    'dock_collapse_violation',
    'dock_hydrophobicity',
}
pose_metrics = {
    # 'entity_max_radius_ratio_v',
    # 'entity_min_radius_ratio_v',
    # 'entity_number_of_residues_ratio_v',
    # 'entity_radius_ratio_v',
    'entity_max_radius_average_deviation',
    'entity_min_radius_average_deviation',
    # 'entity_number_of_residues_average_deviation'
    'entity_radius_average_deviation',
    'interface_b_factor',
    'interface1_secondary_structure_fragment_topology',
    'interface1_secondary_structure_fragment_count',
    'interface1_secondary_structure_topology',
    'interface1_secondary_structure_count',
    'interface2_secondary_structure_fragment_topology',
    'interface2_secondary_structure_fragment_count',
    'interface2_secondary_structure_topology',
    'interface2_secondary_structure_count',
    'interface_secondary_structure_fragment_topology',
    'interface_secondary_structure_fragment_count',
    'interface_secondary_structure_topology',
    'interface_secondary_structure_count',
    'maximum_radius',
    'minimum_radius',
    'multiple_fragment_ratio',
    'nanohedra_score_normalized',
    'nanohedra_score_center_normalized',
    'nanohedra_score',
    'nanohedra_score_center',
    'number_residues_interface_fragment_total',
    'number_residues_interface_fragment_center',
    'number_fragments_interface',
    'number_residues_interface',
    'number_residues_interface_non_fragment',
    'percent_fragment_helix',
    'percent_fragment_strand',
    'percent_fragment_coil',
    'percent_residues_fragment_interface_total',
    'percent_residues_fragment_interface_center',
    'percent_residues_non_fragment_interface',
    'pose_length',
}
fragment_metrics = {
    'interface1_secondary_structure_fragment_topology',
    'interface1_secondary_structure_fragment_count',
    'interface1_secondary_structure_topology',
    'interface1_secondary_structure_count',
    'interface2_secondary_structure_fragment_topology',
    'interface2_secondary_structure_fragment_count',
    'interface2_secondary_structure_topology',
    'interface2_secondary_structure_count',
    'interface_secondary_structure_fragment_topology',
    'interface_secondary_structure_fragment_count',
    'interface_secondary_structure_topology',
    'interface_secondary_structure_count',
    'maximum_radius',
    'minimum_radius',
    'multiple_fragment_ratio',
    'nanohedra_score_normalized',
    'nanohedra_score_center_normalized',
    'nanohedra_score',
    'nanohedra_score_center',
    'number_residues_interface_fragment_total',
    'number_residues_interface_fragment_center',
    'number_fragments_interface',
    'number_residues_interface',
    'number_residues_interface_non_fragment',
    'percent_fragment_helix',
    'percent_fragment_strand',
    'percent_fragment_coil',
    'percent_residues_fragment_interface_total',
    'percent_residues_fragment_interface_center',
    'percent_residues_non_fragment_interface',
}
# These metrics are necessary for all calculations performed during the analysis script.
# They are formatted differently from Rosetta output. If missing, something will fail
rosetta_required = {
    'buried_unsatisfied_hbonds_complex', 'buns1_unbound', 'contact_count', 'coordinate_constraint',
    'favor_residue_energy', 'hbonds_res_selection_complex', 'hbonds_res_selection_1_bound',
    'interface_separation', 'interaction_energy_complex', 'rosetta_reference_energy', 'shape_complementarity',
    # 'interface_connectivity1',
    # 'interface_energy_1_bound', 'interface_energy_1_unbound',  'interface_energy_complex',
    # putils.protocol,
    # 'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
    # 'sasa_hydrophobic_1_bound', 'sasa_polar_1_bound', 'sasa_total_1_bound',
    # 'solvation_energy_complex', 'solvation_energy_1_bound', 'solvation_energy_1_unbound'
    # 'buns2_unbound',
    # 'hbonds_res_selection_2_bound', 'interface_connectivity2',
    # 'interface_energy_2_bound', 'interface_energy_2_unbound',
    # 'sasa_hydrophobic_2_bound', 'sasa_polar_2_bound', 'sasa_total_2_bound',
    # 'solvation_energy_2_bound', 'solvation_energy_2_unbound',
    # 'rmsd'
    # 'buns_asu_hpol', 'buns_nano_hpol', 'buns_asu', 'buns_nano', 'buns_total',
    # 'fsp_total_stability', 'full_stability_complex',
    # 'number_hbonds', 'number_residues_interface',
    # 'average_fragment_z_score', 'nanohedra_score', 'number_fragments_interface',
    # 'interface_b_factor_per_res',
}
columns_to_rename = {'shape_complementarity_median_dist': 'interface_separation',
                     'shape_complementarity_core_median_dist': 'interface_core_separation',
                     'ref': 'rosetta_reference_energy',
                     'interaction_energy_density_filter': 'interaction_energy_per_residue'
                     # 'relax_switch': protocol, 'no_constraint_switch': protocol, 'limit_to_profile_switch': protocol,
                     # 'combo_profile_switch': protocol, 'design_profile_switch': protocol,
                     # 'favor_profile_switch': protocol, 'consensus_design_switch': protocol,
                     # 'interface_energy_res_summary_complex': 'interface_energy_complex',
                     # 'interface_energy_res_summary_1_bound': 'interface_energy_1_bound',
                     # 'interface_energy_res_summary_2_bound': 'interface_energy_2_bound',
                     # 'interface_energy_res_summary_1_unbound': 'interface_energy_1_unbound',
                     # 'interface_energy_res_summary_2_unbound': 'interface_energy_2_unbound',
                     # 'sasa_res_summary_hydrophobic_complex': 'sasa_hydrophobic_complex',
                     # 'sasa_res_summary_polar_complex': 'sasa_polar_complex',
                     # 'sasa_res_summary_total_complex': 'sasa_total_complex',
                     # 'sasa_res_summary_hydrophobic_1_bound': 'sasa_hydrophobic_1_bound',
                     # 'sasa_res_summary_polar_1_bound': 'sasa_polar_1_bound',
                     # 'sasa_res_summary_total_1_bound': 'sasa_total_1_bound',
                     # 'sasa_res_summary_hydrophobic_2_bound': 'sasa_hydrophobic_2_bound',
                     # 'sasa_res_summary_polar_2_bound': 'sasa_polar_2_bound',
                     # 'sasa_res_summary_total_2_bound': 'sasa_total_2_bound',
                     # 'solvation_total_energy_complex': 'solvation_energy_complex',
                     # 'solvation_total_energy_1_bound': 'solvation_energy_1_bound',
                     # 'solvation_total_energy_2_bound': 'solvation_energy_2_bound',
                     # 'solvation_total_energy_1_unbound': 'solvation_energy_1_unbound',
                     # 'solvation_total_energy_2_unbound': 'solvation_energy_2_unbound',
                     # 'R_int_connectivity_1': 'interface_connectivity1',
                     # 'R_int_connectivity_2': 'interface_connectivity2',
                     }
#                      'total_score': 'REU', 'decoy': 'design', 'symmetry_switch': 'symmetry',
clean_up_intermediate_columns = [
    # 'int_energy_no_intra_residue_score',
    # 'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
    # 'sasa_hydrophobic_bound', 'sasa_hydrophobic_1_bound', 'sasa_hydrophobic_2_bound',
    # 'sasa_polar_bound', 'sasa_polar_1_bound', 'sasa_polar_2_bound',
    # 'sasa_total_bound', 'sasa_total_1_bound', 'sasa_total_2_bound',
    'buried_unsatisfied_hbonds_unbound',
    # 'buried_unsatisfied_hbonds_complex', 'buried_unsatisfied_hbonds_unbound1', 'buried_unsatisfied_hbonds_unbound2',
    # 'solvation_energy', 'solvation_energy_complex',
    # 'solvation_energy_1_bound', 'solvation_energy_2_bound', 'solvation_energy_1_unbound',
    # 'solvation_energy_2_unbound',
    # 'interface_energy_1_bound', 'interface_energy_1_unbound', 'interface_energy_2_bound',
    # 'interface_energy_2_unbound',
    # 'interface_solvation_energy_bound',
    # 'interface_solvation_energy_unbound', 'interface_solvation_energy_complex'
]
protocol_specific_columns = ['HBNet_NumUnsatHpol', 'HBNet_Saturation', 'HBNet_Score']
# Some of these are unneeded now, but hanging around in case renaming occurred
unnecessary = ['int_area_asu_hydrophobic', 'int_area_asu_polar', 'int_area_asu_total',
               'int_area_ex_asu_hydrophobic', 'int_area_ex_asu_polar', 'int_area_ex_asu_total',
               'int_energy_context_asu', 'int_energy_context_unbound',
               'int_energy_res_summary_asu', 'int_energy_res_summary_unbound',
               'interaction_energy', 'interaction_energy_asu', 'interaction_energy_oligomerA',
               'interaction_energy_oligomerB', 'interaction_energy_unbound', 'res_type_constraint', 'time', 'REU',
               'full_stability_complex', 'full_stability_oligomer', 'fsp_total_stability',
               'full_stability_1_unbound', 'full_stability_2_unbound',
               'cst_weight', 'fsp_energy', 'int_area_res_summary_hydrophobic_1_unbound',
               'int_area_res_summary_polar_1_unbound', 'int_area_res_summary_total_1_unbound',
               'int_area_res_summary_hydrophobic_2_unbound', 'int_area_res_summary_polar_2_unbound',
               'int_area_res_summary_total_2_unbound', 'int_area_total', 'int_area_polar', 'int_area_hydrophobic',
               'int_energy_context_1_unbound', 'int_energy_res_summary_1_unbound', 'int_energy_context_2_unbound',
               'int_energy_res_summary_2_unbound', 'int_energy_res_summary_complex', 'int_sc', 'int_sc_median_dist',
               # 'solvation_energy_1_bound', 'solvation_energy_2_bound', 'solvation_energy_bound',
               # 'solvation_energy_1_unbound', 'solvation_energy_2_unbound', 'solvation_energy_unbound',
               'hbonds_res_selection_asu', 'hbonds_res_selection_unbound',
               'decoy', 'final_sequence', 'symmetry_switch', 'metrics_symmetry', 'oligomer_switch', 'total_score',
               # 'protocol_switch'
               'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex',
               # 'buns_asu', 'buns_asu_hpol', 'buns_nano', 'buns_nano_hpol', 'buns_total',
               'angle_constraint', 'atom_pair_constraint', 'chainbreak', 'coordinate_constraint', 'dihedral_constraint',
               'metalbinding_constraint', 'rmsd', 'repack_switch', 'sym_status',
               'core_design_residue_count', 'shape_complementarity_core', 'interface_core_separation',
               'interaction_energy_density_filter', 'interaction_energy_density', 'shape_complementarity_hbnet_core',
               'maxsub', 'rms', 'score']
#                'repacking',
# remove_score_columns = ['hbonds_res_selection_asu', 'hbonds_res_selection_unbound']
#                'full_stability_oligomer_A', 'full_stability_oligomer_B']

# columns_to_remove = ['decoy', 'symmetry_switch', 'metrics_symmetry', 'oligomer_switch', 'total_score',
#                      'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex']

# Subtract columns using tuple [0] - [1] to make delta column
rosetta_delta_pairs = {
    'buried_unsatisfied_hbonds': ('buried_unsatisfied_hbonds_complex', 'buried_unsatisfied_hbonds_unbound'),  # Rosetta
    # Replace by Residues summation
    # 'interface_energy': ('interface_energy_complex', 'interface_energy_unbound'),  # Rosetta
    # 'interface_energy_no_intra_residue_score': ('interface_energy_complex', 'interface_energy_bound'),
    'interface_bound_activation_energy': ('interface_energy_bound', 'interface_energy_unbound'),  # Rosetta
    'interface_solvation_energy':
        ('interface_solvation_energy_unbound', 'interface_solvation_energy_complex'),  # Rosetta
    'interface_solvation_energy_activation':
        ('interface_solvation_energy_unbound', 'interface_solvation_energy_bound'),  # Rosetta
    # 'interface_area_hydrophobic': ('sasa_hydrophobic_bound', 'sasa_hydrophobic_complex'),
    # 'interface_area_polar': ('sasa_polar_bound', 'sasa_polar_complex'),
    # 'interface_area_total': ('sasa_total_bound', 'sasa_total_complex')
}
#     'int_energy_context_delta': ('int_energy_context_complex', 'int_energy_context_oligomer'),
#     'full_stability_delta': ('full_stability_complex', 'full_stability_oligomer')}
#     'number_hbonds': ('hbonds_res_selection_complex', 'hbonds_oligomer')}


# divide columns using tuple [0] / [1] to make divide column
rosetta_division_pairs = {
    'buried_unsatisfied_hbond_density': ('buried_unsatisfied_hbonds', 'interface_area_total'),
    'interface_energy_density': ('interface_energy', 'interface_area_total'),
}
division_pairs = {
    'percent_interface_area_hydrophobic': ('interface_area_hydrophobic', 'interface_area_total'),
    'percent_interface_area_polar': ('interface_area_polar', 'interface_area_total'),
    'percent_core': ('core', 'number_residues_interface'),
    'percent_rim': ('rim', 'number_residues_interface'),
    'percent_support': ('support', 'number_residues_interface'),
}  # Rosetta

# All Rosetta based score terms ref is most useful to keep for whole pose to give "unfolded ground state"
rosetta_terms = ['lk_ball_wtd', 'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'yhh_planarity', 'dslf_fa13',
                 'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'ref']

# Current protocols in use in interface_design.xml
rosetta_design_protocols = [
    'design_profile_switch', 'favor_profile_switch', 'limit_to_profile_switch', 'structure_background_switch']
# Todo adapt to any user protocol!
protocols_of_interest = {putils.design_profile, putils.structure_background, putils.hbnet_design_profile}
# protocols_of_interest = ['combo_profile', 'limit_to_profile', 'no_constraint']  # Used for P432 models

protocol_column_types = ['mean', 'sequence_design']  # 'stats',
# Specific columns of interest to distinguish between design trajectories
significance_columns = ['buried_unsatisfied_hbonds',
                        'contact_count', 'interface_energy', 'interface_area_total', 'number_hbonds',
                        'percent_interface_area_hydrophobic', 'shape_complementarity', 'interface_solvation_energy']
# sequence_columns = ['divergence_evolution_per_residue', 'divergence_fragment_per_residue',
#                     'observed_evolution', 'observed_fragment']
multiple_sequence_alignment_dependent_metrics = \
    ['collapse_increase_significance_by_contact_order_z', 'collapse_increased_z', 'collapse_deviation_magnitude',
     'collapse_sequential_peaks_z', 'collapse_sequential_z']
profile_dependent_metrics = ['divergence_evolution_per_residue', 'observed_evolution']
all_evolutionary_metrics = multiple_sequence_alignment_dependent_metrics + profile_dependent_metrics
frag_profile_dependent_metrics = ['divergence_fragment_per_residue', 'observed_fragment']
# per_res_keys = ['jsd', 'des_jsd', 'int_jsd', 'frag_jsd']


def read_scores(file: AnyStr, key: str = 'decoy') -> dict[str, dict[str, str]]:
    """Take a json formatted metrics file and incorporate entries into nested dictionaries with "key" as outer key

    Automatically formats scores according to conventional metric naming scheme, ex: "R_", "S_", or "M_" prefix removal

    Args:
        file: Location on disk of scorefile
        key: Name of the json key to use as outer dictionary identifier
    Returns:
        The parsed scorefile
            Ex {'design_identifier1': {'metric_key': metric_value, ...}, 'design_identifier2': {}, ...}
    """
    with open(file, 'r') as f:
        scores = {}
        for json_entry in f.readlines():
            formatted_scores = {}
            for score, value in loads(json_entry).items():
                if 'res_' in score:  # 'per_res_'):  # There are a lot of these scores in particular
                    formatted_scores[score] = value
                elif score.startswith('R_'):
                    formatted_scores[score.replace('R_', '').replace('S_', '')] = value
                else:
                    # # res_summary replace is used to take sasa_res_summary and other res_summary metrics "string" off
                    # score = score.replace('res_summary_', '')
                    # score = score.replace('res_summary_', '').replace('solvation_total', 'solvation')
                    formatted_scores[columns_to_rename.get(score, score)] = value

            design = formatted_scores.pop(key)
            if design not in scores:
                scores[design] = formatted_scores
            else:
                # # To ensure old trajectories don't have lingering protocol info
                # for protocol in protocols:
                #     if protocol in entry:  # Ensure that the new scores has a protocol before removing the old one.
                #         for rm_protocol in protocols:
                #             scores[design].pop(rm_protocol, None)
                scores[design].update(formatted_scores)

    return scores


def keys_from_trajectory_number(pdb_dict):
    """Remove all string from dictionary keys except for string after last '_'. Ex 'design_0001' -> '0001'

    Returns:
        (dict): {cleaned_key: value, ...}
    """
    return {key.split('_')[-1]: value for key, value in pdb_dict.items()}


def join_columns(row):  # UNUSED
    """Combine columns in a dataframe with the same column name. Keep only the last column record

    Returns:
        (str): The column name
    """
    new_data = ','.join(row[row.notnull()].astype(str))
    return new_data.split(',')[-1]


def columns_to_new_column(df: pd.DataFrame, columns: dict[str, tuple[str, ...]], mode: str = 'add'):
    """Set new column value by taking an operation of one column on another

    Can perform summation and subtraction if a set of columns is provided
    Args:
        df: Dataframe where the columns are located
        columns: Keys are new column names, values are tuple of existing columns where
            df[key] = value[0] mode(operation) value[1] mode(operation) ...
        mode: What operator to use?
            Viable options are included in the operator module {'sub', 'mul', 'truediv', ...}
    Returns:
        Dataframe with new column values
    """
    for new_column, column_set in columns.items():
        try:  # Todo check why using attrgetter(mode)(operator) ?
            df[new_column] = operator.attrgetter(mode)(operator)(df[column_set[0]], df[column_set[1]])
        except KeyError:
            pass
        except IndexError:
            raise IndexError(f'The number of columns in the set {column_set} is not >= 2. {new_column} not possible!')
        if len(column_set) > 2 and mode in ['add', 'sub']:  # >2 values in set, perform repeated operations Ex: SUM, SUB
            for extra_column in column_set[2:]:  # perform an iteration for every N-2 items in the column_set
                try:
                    df[new_column] = operator.attrgetter(mode)(operator)(df[new_column], df[extra_column])
                except KeyError:
                    pass

    return df


def hbond_processing(design_scores: dict, columns: list[str]) -> dict[str, set]:
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering
    Args:
        design_scores: {'001': {'buns': 2.0, 'per_res_energy_complex_15A': -2.71, ...,
                                'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...',
                                'hbonds_res_selection_1_bound': '26A'}, ...}
        columns : ['hbonds_res_selection_complex', 'hbonds_res_selection_1_unbound',
                   'hbonds_res_selection_2_unbound']
    Returns:
        {'0001': {34, 54, 67, 68, 106, 178}, ...}
    """
    hbonds = {}
    for design, scores in design_scores.items():
        unbound_bonds, complex_bonds = set(), set()
        for column in columns:
            if column not in scores:
                continue
            meta_data = column.split('_')  # ['hbonds', 'res', 'selection', 'complex/interface_number', '[unbound]']
            parsed_hbonds = set(int(hbond.translate(utils.keep_digit_table))
                                for hbond in scores.get(column, '').split(',') if hbond != '')  # check if '' in case no hbonds
            if meta_data[3] == 'complex':
                complex_bonds = parsed_hbonds
            else:  # from another state
                unbound_bonds = unbound_bonds.union(parsed_hbonds)
        if complex_bonds:  # 'complex', '1', '2'
            hbonds[design] = complex_bonds.difference(unbound_bonds)
            # hbonds[entry] = [hbonds_entry['complex'].difference(hbonds_entry['1']).difference(hbonds_entry['2']))]
            #                                                         hbonds_entry['A']).difference(hbonds_entry['B'])
        else:  # no hbonds were found in the complex
            hbonds[design] = complex_bonds
            # logger.error('%s: Missing hbonds_res_selection_ data for %s. Hbonds inaccurate!' % (pose, entry))

    return hbonds


def dirty_hbond_processing(design_scores: dict) -> dict[str, set]:
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering
    Args:
        design_scores: {'001': {'buns': 2.0, 'per_res_energy_complex_15A': -2.71, ...,
                                'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...',
                                'hbonds_res_selection_1_bound': '26A'}, ...}
    Returns:
        {'001': {34, 54, 67, 68, 106, 178}, ...}
    """
    hbonds = {}
    for design, scores in design_scores.items():
        unbound_bonds, complex_bonds = set(), set()
        for column, value in scores.items():
            if 'hbonds_res_' not in column:  # if not column.startswith('hbonds_res_selection'):
                continue
            meta_data = column.split('_')  # ['hbonds', 'res', 'selection', 'complex/interface_number', '[unbound]']
            parsed_hbonds = set(int(hbond.translate(utils.keep_digit_table))
                                for hbond in value.split(',') if hbond != '')  # check if '' in case no hbonds
            # if meta_data[-1] == 'bound' and offset:  # find offset according to chain
            #     res_offset = offset[meta_data[-2]]
            #     parsed_hbonds = set(residue + res_offset for residue in parsed_hbonds)
            if meta_data[3] == 'complex':
                complex_bonds = parsed_hbonds
            else:  # From another state
                unbound_bonds = unbound_bonds.union(parsed_hbonds)
        if complex_bonds:  # 'complex', '1', '2'
            hbonds[design] = complex_bonds.difference(unbound_bonds)
            # hbonds[design] = [hbonds_entry['complex'].difference(hbonds_entry['1']).difference(hbonds_entry['2']))]
            # #                                                       hbonds_entry['A']).difference(hbonds_entry['B']))]
        else:  # no hbonds were found in the complex
            hbonds[design] = complex_bonds
            # logger.error('%s: Missing hbonds_res_selection_ scores for %s. Hbonds inaccurate!' % (pose, design))

    return hbonds


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


def interface_composition_similarity(series: Mapping) -> float:
    """Calculate the composition difference for pose residue classification

    Args:
        series: Mapping from 'interface_area_total', 'core', 'rim', and 'support' to values
    Returns:
        Average similarity for expected residue classification given the observed classification
    """
    # Calculate modelled number of residues according to buried surface area (Levy, E 2010)
    def core_res_fn(bsa):
        return 0.01 * bsa + 0.6

    def rim_res_fn(bsa):
        return 0.01 * bsa - 2.5

    def support_res_fn(bsa):
        return 0.006 * bsa + 5

    classification_fxn_d = {'core': core_res_fn, 'rim': rim_res_fn, 'support': support_res_fn}

    int_area = series['interface_area_total']  # buried surface area
    if int_area <= 250:
        return np.nan

    class_ratio_differences = []
    for residue_class, function in classification_fxn_d.items():
        expected = function(int_area)
        class_ratio_difference = (1 - (abs(series[residue_class] - expected) / expected))
        if class_ratio_difference < 0:
            # Above calculation fails to bound between 0 and 1 with large obs values due to proportion > 1
            class_ratio_difference = 0
        class_ratio_differences.append(class_ratio_difference)

    return sum(class_ratio_differences) / len(class_ratio_differences)


def incorporate_sequence_info(design_residue_scores: dict[str, dict], sequences: dict[str, Sequence[str]]) \
        -> dict[str, dict]:
    """Incorporate mutation measurements into residue info. design_residue_scores and mutations must be the same index

    Args:
        design_residue_scores: {'001': {15: {'complex': -2.71, 'bound': [-1.9, 0], 'unbound': [-1.9, 0],
                                             'solv_complex': -2.71, 'solv_bound': [-1.9, 0], 'solv_unbound': [-1.9, 0],
                                             'fsp': 0., 'cst': 0.}, ...}, ...}
        sequences: {'001': 'MKDLSAVLIRLAD...', '002': '', ...}
    Returns:
        {'001': {15: {'type': 'T', 'energy_delta': -2.71, 'coordinate_constraint': 0. 'residue_favored': 0., 'hbond': 0}
                 ...}, ...}
    """
    # warn = False
    # reference_data = mutations.get(putils.reference_name)
    # pose_length = len(reference_data)
    for design, residue_info in design_residue_scores.items():
        sequence = sequences.get(design)
        # mutation_data = mutations.get(design)
        # if not mutation_data:
        #     continue

        # remove_residues = []
        for residue_index, data in residue_info.items():
            data['type'] = sequence[residue_index]
            # try:  # Set residue AA type based on provided mutations
            #     data['type'] = mutation_data[residue_index]
            # except KeyError:  # Residue is not in mutations, probably missing as it is not a mutation
            #     try:  # Fill in with AA from putils.reference_name seq
            #         data['type'] = reference_data[residue_index]
            #     except KeyError:  # Residue is out of bounds on pose length
            #         # Possibly a virtual residue or string that was processed incorrectly from the keep_digit_table
            #         if not warn:
            #             logger.error(f'Encountered residue index "{residue_index}" which is not within the pose size '
            #                          f'"{pose_length}" and will be removed from processing. This is likely an error '
            #                          f'with residue processing or residue selection in the specified rosetta protocol.'
            #                          f' If there were warnings produced indicating a larger residue number than pose '
            #                          f'size, this problem was not addressable heuristically and something else has '
            #                          f'occurred. It is likely that this residue number is not useful if you indeed have'
            #                          f' output_as_pdb_nums="true"')
            #             warn = True
            #         remove_residues.append(residue_index)
            #         continue

        # # Clean up any incorrect residues
        # for residue in remove_residues:
        #     residue_info.pop(residue)

    return design_residue_scores


def process_residue_info(design_residue_scores: dict, hbonds: dict = None) -> dict:
    """Process energy metrics to Pose formatted dictionary from multiple measurements per residue
    and incorporate hydrogen bond information. design_residue_scores and hbonds must be the same index

    Args:
        design_residue_scores: {'001': {15: {'complex': -2.71, 'bound': [-1.9, 0], 'unbound': [-1.9, 0],
                                             'solv_complex': -2.71, 'solv_bound': [-1.9, 0], 'solv_unbound': [-1.9, 0],
                                             'fsp': 0., 'cst': 0.}, ...}, ...}
        hbonds: {'001': [34, 54, 67, 68, 106, 178], ...}
    Returns:
        {'001': {15: {'type': 'T', 'energy_delta': -2.71, 'coordinate_constraint': 0. 'residue_favored': 0., 'hbond': 0}
                 ...}, ...}
    """
    if hbonds is None:
        hbonds = {}

    for design, residue_info in design_residue_scores.items():
        design_hbonds = hbonds.get(design, [])
        for residue_number, data in residue_info.items():
            # Set hbond bool if available
            data['hbond'] = 1 if residue_number in design_hbonds else 0
            # Compute the energy delta which requires summing the unbound energies
            data['unbound'] = sum(data['unbound'])
            data['energy_delta'] = data['complex'] - data['unbound']
            # Compute the "preconfiguration" energy delta which requires summing the bound energies
            data['bound'] = sum(data['bound'])
            # data['energy_bound_activation'] = data['bound'] - data['unbound']
            data['solv_bound'] = sum(data['solv_bound'])
            data['solv_unbound'] = sum(data['solv_unbound'])
            data['coordinate_constraint'] = data.get('cst', 0.)
            data['residue_favored'] = data.get('fsp', 0.)
            # if residue_data[residue_number]['energy'] <= hot_spot_energy:
            #     residue_data[residue_number]['hot_spot'] = 1

    return design_residue_scores


def collapse_per_residue(sequence_groups: Iterable[Iterable[Sequence[str]]],
                         residue_contact_order_z: np.ndarray, reference_collapse: np.ndarray, **kwargs) \
        -> list[dict[str, float]]:
    # collapse_profile: np.ndarray = None,
    # reference_mean: float | np.ndarray = None,
    # reference_std: float | np.ndarray = None,
    """Measure per-residue sequence folding metrics based on reference values including contact order z score and
    hydrophobic collapse

    Args:
        sequence_groups: Groups of sequences, where the outer nest is each sample and the inner nest are unique polymers
        residue_contact_order_z: The per-residue contact order z score from a reference structure
        reference_collapse: The per-residue hydrophobic collapse values measured from a reference sequence
    Keyword Args:
        hydrophobicity: int = 'standard' – The hydrophobicity scale to consider. Either 'standard' (FILV),
            'expanded' (FMILYVW), or provide one with 'custom' keyword argument
        custom: mapping[str, float | int] = None – A user defined mapping of amino acid type, hydrophobicity value pairs
        alphabet_type: alphabet_types = None – The amino acid alphabet if the sequence consists of integer characters
        lower_window: int = 3 – The smallest window used to measure
        upper_window: int = 9 – The largest window used to measure
    Returns:
        The mapping of collapse metric to per-residue values for the concatenated sequence in each sequence_groups.
            These include:
            {'collapse_deviation_magnitude',
             'collapse_increase_significance_by_contact_order_z',
             'collapse_increased_z',
             'collapse_new_positions',
             'collapse_new_position_significance',
             'collapse_sequential_peaks_z',
             'collapse_sequential_z',
             'collapse_significance_by_contact_order_z',
             'hydrophobic_collapse'
             }
    """
    #    collapse_profile: The per-residue hydrophobic collapse values measured from a reference SequenceProfile
    #    reference_mean: The hydrophobic collapse mean value(s) to use as a reference for z-score calculation
    #    reference_std: The hydrophobic collapse deviation value(s) to use as a reference for z-score calculation
    hydrophobicity = kwargs.get('hydrophobicity')
    if not hydrophobicity:  # Set to the standard
        hydrophobicity = kwargs['hydrophobicity'] = 'standard'
    # else:
    #     if hydrophobicity != 'standard':
    #         logger.warning(f'Found hydrophobicity="{hydrophobicity}". This is incompatible without passing '
    #                        'reference_mean/_std. Setting hydrophobicity="standard"')
    #         kwargs['hydrophobicity'] = 'standard'

    significance_threshold = collapse_thresholds[hydrophobicity]
    # # if collapse_profile is not None and collapse_profile.size:  # Not equal to zero
    # if reference_mean is None or reference_std is None:
    #     reference_mean = significance_threshold
    #     reference_std = collapse_reported_std
    # # else:
    # #     reference_mean = np.nanmean(collapse_profile, axis=-2)
    # #     reference_std = np.nanstd(collapse_profile, axis=-2)
    # #     # Use only the reference (index=0) hydrophobic_collapse_index to calculate a reference collapse z-score
    # #     reference_collapse_z_score = utils.z_score(collapse_profile[0], reference_mean, reference_std)
    # #     reference_collapse_bool = reference_mean > collapse_significance_threshold

    # reference_collapse_bool = np.where(reference_collapse > collapse_significance_threshold, 1, 0)
    # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
    reference_collapse_bool = (reference_collapse > significance_threshold).astype(int)
    # [False, False, False, False, True, True, False, False, True, True, ...]
    # reference_collapse_z_score = utils.z_score(reference_collapse, reference_mean, reference_std)
    reference_collapse_z_score = z_score(reference_collapse, significance_threshold, collapse_reported_std)

    # Linearly weight residue by sequence position (early > late) with the halfway position (midpoint) at .5
    # midpoint = .5
    scale = 1  # / midpoint
    folding_and_collapse = []
    # for pose_idx, pose in enumerate(poses_of_interest):
    #     collapse = []
    #     for entity_idx, entity in enumerate(pose.entities):
    #         sequence_length = entity.number_of_residues
    #         collapse.append(entity.hydrophobic_collapse())
    for pose_idx, sequences in enumerate(sequence_groups):
        # Gather all the collapse info for the particular sequence group
        collapse = np.concatenate([hydrophobic_collapse_index(sequence, **kwargs)
                                   for entity_idx, sequence in enumerate(sequences)])
        # Scale the collapse by the standard collapse threshold and make z score
        # collapse_z = utils.z_score(collapse, reference_mean, reference_std)
        collapse_z = z_score(collapse, significance_threshold, collapse_reported_std)
        # Find the difference between the sequence and the reference
        difference_collapse_z = collapse_z - reference_collapse_z_score
        # The sum of all sequence regions z-scores experiencing increased collapse. Measures the normalized
        # magnitude of additional hydrophobic collapse
        # collapse_deviation_magnitude_sum = np.abs(difference_collapse_z).sum()
        collapse_deviation_magnitude = np.abs(difference_collapse_z)

        # Find the indices where the sequence collapse has increased compared to reference collapse_profile
        increased_collapse_z = np.maximum(difference_collapse_z, 0)
        # collapse_increased_z_sum = increased_collapse_z.sum()

        # Sum the contact order, scaled proportionally by the collapse increase. More negative is more isolated
        # collapse. Positive indicates poor maintaning of the starting collapse
        # collapse_increase_significance_by_contact_order_z_sum = \
        #     np.sum(residue_contact_order_z * increased_collapse_z)
        # collapse_increase_significance_by_contact_order_z = residue_contact_order_z * increased_collapse_z

        # Where collapse is occurring
        collapsing_positions_z = np.maximum(collapse_z, 0)
        # ^ [0, 0, 0, 0, 0.04, 0.06, 0, 0, 0.1, 0.07, ...]
        collapse_bool = collapsing_positions_z != 0  # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
        # Check if increased collapse positions resulted in a location of "new collapse"
        # i.e. sites where collapse occurs compared to reference
        new_collapsing = (collapse_bool - reference_collapse_bool) == 1
        # Ex, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ...]

        # Calculate "islands". Ensure position is collapsing, while the reference has no collapsing neighbors
        # list is faster to index than np.ndarray i.e. new_collapse = np.zeros_like(collapse_bool)
        new_collapse = [True if collapse and (not ref_minus1 and not ref_plus1) else False
                        for ref_minus1, collapse, ref_plus1 in
                        # Trim the sequence to a 3 residue window (-1, 0, 1)
                        zip(reference_collapse[:-2].tolist(),
                            new_collapsing[1:-1].tolist(),
                            reference_collapse[2:].tolist())]
        # Finish by calculating first and last indices as well and combining
        new_collapse = [True if new_collapsing[0] and not reference_collapse[1] else False] \
            + new_collapse \
            + [True if new_collapsing[-1] and not reference_collapse[-2] else False]

        # Find new collapse positions
        new_collapse_peak_start = [0 for _ in range(len(collapse_bool))]  # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...]
        # Keep track of how many discrete collapsing segments exist and where their boundaries are
        collapse_peak_start = new_collapse_peak_start.copy()  # [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, ...]
        sequential_collapse_points = np.zeros_like(collapse_bool)  # [-1, -1, -1, -1, 0, 0, 0, 0, 1, 1, ...]
        collapse_iterator = -1  # Start at -1 so that the first point eventually is equal to a 0 subtraction. Was 0
        for prior_idx, idx in enumerate(range(1, len(collapse_z))):
            # Compare neighboring residues in the new_collapse and collapse_peak_start
            # Both conditions are only True when 0 -> 1 transition occurs
            if new_collapse[prior_idx] < new_collapse[idx]:
                new_collapse_peak_start[idx] = 1
            if collapse_bool[prior_idx] < collapse_bool[idx]:
                collapse_peak_start[idx] = 1
                collapse_iterator += 1
            sequential_collapse_points[idx] = collapse_iterator

        # if collapse_profile is not None and collapse_profile.size:  # Not equal to zero
        # Compare the measured collapse to the metrics gathered from the collapse_profile
        # # _collapse_z = utils.z_score(standardized_collapse, collapse_profile_mean, collapse_profile_std)
        # _collapse_z = utils.z_score(collapse, reference_mean, reference_std)
        # Find the indices where the _collapse_z is increased versus the reference_collapse_z_score

        try:
            step = 1 / sum(collapse_peak_start)  # This is 1 over the "total_collapse_points"
        except ZeroDivisionError:  # No collapse peaks
            step = 1
        # # Make array for small adjustment to account for first value equal to scale
        # # add_step_array = collapse_bool * step
        # v [1.1, 1.1, 1.1, 1.1, 1, 1, 1, 1, .9, .9, ...]
        sequential_collapse_weights = scale * (1 - step*sequential_collapse_points)
        # Make sequential_collapse_weights only useful at points where collapse increased (i.e. collapse_bool is 1)
        # v [0, 0, 0, 0, 1, 1, 0, 0, .9, .9, ...]
        sequential_collapse_weights *= collapse_bool
        # collapse_sequential_peaks_z_sum = np.sum(sequential_collapse_weights * increased_collapse_z)
        collapse_sequential_peaks_z = sequential_collapse_weights * increased_collapse_z
        # v [1, .99, .98, .97, .96, ...]
        sequence_length = len(collapse)
        sequential_weights = scale * (1 - np.arange(sequence_length)/sequence_length)
        # collapse_sequential_z_sum = np.sum(sequential_weights * increased_collapse_z)
        collapse_sequential_z = sequential_weights * increased_collapse_z
        # else:
        #     # For per-residue
        #     collapse_increase_significance_by_contact_order_z = increased_collapse_z = \
        #         collapse_deviation_magnitude = collapse_sequential_peaks_z = collapse_sequential_z = \
        #         np.zeros_like(collapse)
        #     # # For summing
        #     # collapse_deviation_magnitude_sum = collapse_increase_significance_by_contact_order_z_sum = \
        #     #     collapse_sequential_peaks_z_sum = collapse_sequential_z_sum = collapse_increased_z_sum = 0.

        # Negating inverts contact order z-score to weight high contact order negatively
        residue_contact_order_inverted_z = residue_contact_order_z * -1

        # With 'collapse_new_position_significance'
        #  Use contact order z score and hci to understand designability of an area and its folding modification
        #  For positions experiencing collapse, multiply by inverted contact order
        collapse_significance = residue_contact_order_inverted_z * collapsing_positions_z
        #  Positive values indicate collapse in areas with low contact order
        #  Negative, collapse in high contact order
        #  Indicates the degree to which low contact order segments (+) may be reliant on collapse for folding,
        #  while high contact order (-) may use collapse
        # residue_contact_order_inverted_z = [-1.0, -0.4, 0.8, 0.2, -1.3, -0.2, 0.9, -1.7, ...]
        # collapsing_positions_z = [0, 0, 0, 0, 0.04, 0.06, 0, 0, 0.1, 0.07, ...]

        # Add the concatenated collapse metrics to total
        folding_and_collapse.append({
            'hydrophobic_collapse': collapse,
            'collapse_deviation_magnitude': collapse_deviation_magnitude,
            'collapse_increase_significance_by_contact_order_z':
                residue_contact_order_inverted_z * increased_collapse_z,
            'collapse_increased_z': increased_collapse_z,
            'collapse_new_positions': new_collapse_peak_start,
            'collapse_new_position_significance': new_collapse_peak_start * collapse_significance,
            'collapse_sequential_peaks_z': collapse_sequential_peaks_z,
            'collapse_sequential_z': collapse_sequential_z,
            'collapse_significance_by_contact_order_z': collapse_significance
        })
    return folding_and_collapse


def mutation_conserved(residue_info: dict, bkgnd: dict) -> dict:
    """Process residue mutations compared to evolutionary background. Returns 1 if residue is observed in background

    Both residue_dict and background must be same index
    Args:
        residue_info: {15: {'type': 'T', ...}, ...}
        bkgnd: {0: {'A': 0, 'R': 0, ...}, ...}
    Returns:
        conservation_dict: {15: 1, 21: 0, 25: 1, ...}
    """
    return {res: 1 if bkgnd[res][info['type']] > 0 else 0 for res, info in residue_info.items() if res in bkgnd}
    # return [1 if bgd[info['type']] > 0 else 0 for info, bgd in zip(residue_info, bkgnd)]


def per_res_metric(sequence_metrics: dict[Any, float] | dict[Any, dict[str, float]], key: str = None) -> float:
    """Find metric value average over all residues in a per residue dictionary with metric specified by key

    Args:
        sequence_metrics: {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
        key: Name of the metric to average
    Returns:
        The average metric 0.367
    """
    s, total = 0.0, 0
    if key:
        for residue_metrics in sequence_metrics.values():
            value = residue_metrics.get(key)
            if value:
                s += value
                total += 1
    else:
        for total, residue_metric in enumerate(sequence_metrics.values(), 1):
            s += residue_metric

    if total == 0:
        return 0.
    else:
        return s / total


def calculate_residue_surface_area(per_residue_df: pd.DataFrame) -> pd.DataFrame:
    #  index_residues: list[int] = slice(None, None)
    """From a DataFrame with per residue values, tabulate the values relating to interfacial surface area

    Args:
        per_residue_df: The DataFrame with MultiIndex columns where level1=residue_numbers, level0=residue_metric
    Returns:
        The same dataframe with added columns
    """
    # Make buried surface area (bsa) columns
    bound_hydro = per_residue_df.loc[:, idx_slice[:, 'sasa_hydrophobic_bound']]
    bound_polar = per_residue_df.loc[:, idx_slice[:, 'sasa_polar_bound']]
    complex_hydro = per_residue_df.loc[:, idx_slice[:, 'sasa_hydrophobic_complex']]
    complex_polar = per_residue_df.loc[:, idx_slice[:, 'sasa_polar_complex']]

    bsa_hydrophobic = (bound_hydro.rename(columns={'sasa_hydrophobic_bound': 'bsa_hydrophobic'})
                       - complex_hydro.rename(columns={'sasa_hydrophobic_complex': 'bsa_hydrophobic'}))
    bsa_polar = (bound_polar.rename(columns={'sasa_polar_bound': 'bsa_polar'})
                 - complex_polar.rename(columns={'sasa_polar_complex': 'bsa_polar'}))
    bsa_total = (bsa_hydrophobic.rename(columns={'bsa_hydrophobic': 'bsa_total'})
                 + bsa_polar.rename(columns={'bsa_polar': 'bsa_total'}))

    # Make sasa_complex_total columns
    bound_total = (bound_hydro.rename(columns={'sasa_hydrophobic_bound': 'sasa_total_bound'})
                   + bound_polar.rename(columns={'sasa_polar_bound': 'sasa_total_bound'}))
    complex_total = (complex_hydro.rename(columns={'sasa_hydrophobic_complex': 'sasa_total_complex'})
                     + complex_polar.rename(columns={'sasa_polar_complex': 'sasa_total_complex'}))

    # Find the relative sasa of the complex and the unbound fraction
    rim_core_support = (bsa_total > bsa_tolerance).to_numpy()
    interior_surface = ~rim_core_support
    # surface_or_rim = per_residue_df.loc[:, idx_slice[index_residues, 'sasa_relative_complex']] > 0.25
    # v These could also be support
    core_or_support_or_interior = per_residue_df.loc[:, idx_slice[:, 'sasa_relative_complex']] < 0.25
    surface_or_rim = ~core_or_support_or_interior
    support_or_interior_not_core_or_rim = per_residue_df.loc[:, idx_slice[:, 'sasa_relative_bound']] < 0.25
    # ^ These could be interior too
    # core_sufficient = np.logical_and(core_or_support_or_interior, rim_core_support).to_numpy()
    interior_residues = np.logical_and(core_or_support_or_interior, interior_surface).rename(
        columns={'sasa_relative_complex': 'interior'})
    surface_residues = np.logical_and(surface_or_rim, interior_surface).rename(
        columns={'sasa_relative_complex': 'surface'})

    support_residues = np.logical_and(support_or_interior_not_core_or_rim, rim_core_support).rename(
        columns={'sasa_relative_bound': 'support'})
    rim_residues = np.logical_and(surface_or_rim, rim_core_support).rename(
        columns={'sasa_relative_complex': 'rim'})
    core_residues = np.logical_and(~support_residues,
                                   np.logical_and(core_or_support_or_interior, rim_core_support).to_numpy()).rename(
        columns={'support': 'core'})

    per_residue_df = per_residue_df.join([bsa_hydrophobic, bsa_polar, bsa_total, bound_total, complex_total,
                                          core_residues, interior_residues, support_residues, rim_residues,
                                          surface_residues
                                          ])
    # Perhaps I need to drop
    per_residue_df.drop(relative_sasa_states, axis=1, level=-1, errors='ignore', inplace=True)
    # per_residue_df = pd.concat([per_residue_df, core_residues, interior_residues, support_residues, rim_residues,
    #                             surface_residues], axis=1)
    return per_residue_df


def sum_per_residue_metrics(df: pd.DataFrame, rename_columns: Mapping[str, str] = None,
                            mean_metrics: Sequence[str] = None) -> pd.DataFrame:
    """From a DataFrame with per-residue values (i.e. a metric in level -1), tabulate all values across each residue

    Renames specific values relating to interfacial energy and solvation energy

    Args:
        df: The DataFrame with MultiIndex columns where level -1 = metric
        rename_columns: Columns to rename as a result of the summation
        mean_metrics: Columns to take the mean instead of the sum
    Returns:
        A new DataFrame with the summation of each metric from all residue_numbers in the per_residue columns
    """
    # # Drop unused particular residues_df columns that have been summed
    # per_residue_drop_columns = per_residue_energy_states + energy_metric_names + per_residue_sasa_states \
    #                            + collapse_metrics + residue_classification \
    #                            + ['errat_deviation', 'hydrophobic_collapse', 'contact_order'] \
    #                            + ['hbond', 'evolution', 'fragment', 'type'] + ['surface', 'interior']
    # residues_df = residues_df.drop(
    #     list(residues_df.loc[:, idx_slice[:, per_residue_drop_columns]].columns),
    #     errors='ignore', axis=1)

    # Group by the columns according to the metrics (level=-1). Upper level(s) are residue identifiers
    groupby_df = df.groupby(axis=1, level=-1)
    rename_columns = {
        'hydrophobic_collapse': 'hydrophobicity',
        # Currently dropped in sum_per_residue_metrics()
        # 'sasa_relative_bound': 'relative_area_bound',
        # 'sasa_relative_complex': 'relative_area_complex',
        **energy_metrics_rename_mapping,
        **sasa_metrics_rename_mapping,
        **renamed_design_metrics,
        **(rename_columns or {})}
    count_df = groupby_df.count().rename(columns=rename_columns)
    # Using min_count=1, we ensure that those columns with np.nan remain np.nan
    summed_df = groupby_df.sum(min_count=1).rename(columns=rename_columns)
    # logger.debug('After residues sum: {summed_df}')
    # _mean_metrics = ['hydrophobicity', 'relative_area_bound', 'relative_area_complex']
    if mean_metrics is None:
        # # Set empty slice
        # mean_metrics = slice(None)
        pass
    else:  # if mean_metrics is not None:  # Make a list and incorporate
        # _mean_metrics += list(mean_metrics)
        summed_df[mean_metrics] = summed_df[mean_metrics].div(count_df[mean_metrics], axis=0)

    return summed_df


def calculate_sequence_observations_and_divergence(alignment: sequence.MultipleSequenceAlignment,
                                                   backgrounds: dict[str, np.ndarray]) \
        -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    #                                                select_indices: list[int] = None) \
    """Gather the observed frequencies from each sequence in a MultipleSequenceAlignment"""
    # mutation_frequencies = pose_alignment.frequencies[[residue-1 for residue in pose.interface_design_residue_numbers]]
    # mutation_frequencies = filter_dictionary_keys(pose_alignment.frequencies, pose.interface_design_residue_numbers)
    # mutation_frequencies = filter_dictionary_keys(pose_alignment['frequencies'], interface_residue_numbers)

    # Calculate amino acid observation percent from residue_info and background SSM's
    # observation_d = {profile: {design: mutation_conserved(info, background)
    #                            for design, numerical_sequence in residue_info.items()}
    # observation_d = {profile: {design: np.where(background[:, numerical_sequence] > 0, 1, 0)
    #                            for design, numerical_sequence in zip(pose_sequences,
    #                                                                  list(pose_alignment.numerical_alignment))}
    #                  for profile, background in profile_background.items()}
    # Find the observed background for each profile, for each designed pose
    # pose_observed_bkd = {profile: {design: freq.mean() for design, freq in design_obs_freqs.items()}
    #                      for profile, design_obs_freqs in observation_d.items()}
    # for profile, observed_frequencies in pose_observed_bkd.items():
    #     scores_df[f'observed_{profile}'] = pd.Series(observed_frequencies)
    # for profile, design_obs_freqs in observation_d.items():
    #     scores_df[f'observed_{profile}'] = \
    #         pd.Series({design: freq.mean() for design, freq in design_obs_freqs.items()})
    # observed_dfs = []
    transposed_alignment = alignment.numerical_alignment.T
    # observed = {profile: np.take_along_axis(background, transposed_alignment, axis=1).T
    observed = {profile: np.where(np.take_along_axis(background, transposed_alignment, axis=1) > 0, 1, 0).T
                for profile, background in backgrounds.items()}
    # for profile, background in profile_background.items():
    #     observed[profile] = np.where(np.take_along_axis(background, transposed_alignment, axis=1) > 0, 1, 0).T
    #     # obs_df = pd.DataFrame(data=np.where(np.take_along_axis(background, transposed_alignment, axis=1) > 0,
    #     #                                     1, 0).T,
    #     #                       index=pose_sequences,
    #     #                       columns=pd.MultiIndex.from_product([residue_indices, [f'observed_{profile}']]))
    #     # observed_dfs.append(obs_df)

    # Calculate Jensen Shannon Divergence using different SSM occurrence data and design mutations
    #                                              both mut_freq and profile_background[profile] are one-indexed
    divergence = {f'divergence_{profile}':
                  # position_specific_jsd(pose_alignment.frequencies, background)
                  position_specific_divergence(alignment.frequencies, background)  # [select_indices]
                  for profile, background in backgrounds.items()}

    return observed, divergence


# def position_specific_jsd(msa: dict[int, dict[str, float]], background: dict[int, dict[str, float]]) -> \
#         dict[int, float]:
#     """Generate the Jensen-Shannon Divergence for a dictionary of residues versus a specific background frequency
#
#     Both msa and background must be the same index
#     Args:
#         msa: {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
#         background: {0: {'A': 0, 'R': 0, ...}, 1: {}, ...}
#             Containing residue index with inner dictionary of single amino acid types
#     Returns:
#         divergence_dict: {15: 0.732, 16: 0.552, ...}
#     """
#     return {idx: js_divergence(freq, background[idx]) for idx, freq in msa.items() if idx in background}
#
#
# def js_divergence(frequencies: dict[str, float], bgd_frequencies: dict[str, float], lambda_: float = 0.5) -> \
#         float:
#     """Calculate residue specific Jensen-Shannon Divergence value
#
#     Args:
#         frequencies: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}
#         bgd_frequencies: {'A': 0, 'R': 0, ...}
#         lambda_: Value bounded between 0 and 1 to calculate the contribution from the observation versus the background
#     Returns:
#         Bounded between 0 and 1. 1 is more divergent from background frequencies
#     """
#     sum_prob1, sum_prob2 = 0, 0
#     for item, frequency in frequencies.items():
#         bgd_frequency = bgd_frequencies.get(item)
#         try:
#             r = (lambda_ * frequency) + ((1 - lambda_) * bgd_frequency)
#         except TypeError:  # bgd_frequency is None, therefore the frequencies can't be compared. Should error be raised?
#             continue
#         try:
#             with warnings.catch_warnings():
#                 # Cause all warnings to always be ignored
#                 warnings.simplefilter('ignore')
#                 try:
#                     prob2 = (bgd_frequency * log(bgd_frequency / r, 2))
#                     sum_prob2 += prob2
#                 except (ValueError, RuntimeWarning):  # math DomainError doesn't raise, instead RunTimeWarn
#                     pass  # continue
#                 try:
#                     prob1 = (frequency * log(frequency / r, 2))
#                     sum_prob1 += prob1
#                 except (ValueError, RuntimeWarning):  # math domain error
#                     continue
#         except ZeroDivisionError:  # r = 0
#             continue
#
#     return lambda_ * sum_prob1 + (1 - lambda_) * sum_prob2


def jensen_shannon_divergence(sequence_frequencies: np.ndarray, background_aa_freq: np.ndarray, **kwargs) -> np.ndarray:
    """Calculate Jensen-Shannon Divergence value for all residues against a background frequency dict

    Args:
        sequence_frequencies: [[0.05, 0.001, 0.1, ...], ...]
        background_aa_freq: [0.11, 0.03, 0.53, ...]
    Keyword Args:
        lambda_: float = 0.5 - Bounded between 0 and 1 indicates weight of the observation versus the background
    Returns:
        The divergence per residue bounded between 0 and 1. 1 is more divergent from background, i.e. [0.732, ...]
    """
    return np.array([js_divergence(sequence_frequencies[idx], background_aa_freq, **kwargs)
                     for idx in range(len(sequence_frequencies))])


def position_specific_jsd(msa: np.ndarray, background: np.ndarray, **kwargs) -> np.ndarray:
    """Generate the Jensen-Shannon Divergence for a dictionary of residues versus a specific background frequency

    Both msa and background must be the same index

    Args:
        msa: {15: {'A': 0.05, 'C': 0.001, 'D': 0.1, ...}, 16: {}, ...}
        background: {0: {'A': 0, 'R': 0, ...}, 1: {}, ...}
            Containing residue index with inner dictionary of single amino acid types
    Keyword Args:
        lambda_: float = 0.5 - Bounded between 0 and 1 indicates weight of the observation versus the background
    Returns:
        The divergence values per position, i.e [0.732, 0.552, ...]
    """
    return np.array([js_divergence(msa[idx], background[idx], **kwargs) for idx in range(len(msa))])


# KL divergence is similar to cross entropy loss or the "log loss"
# divergence of p from q where p is the true distribution and q is the model
# Cross-Entropy = -SUMi->N(probability(pi) * log(probability(qi)))
# Kullback–Leibler-Divergence = -SUMi->N(probability(pi) * log(probability(qi)/probability(pi)))
# Shannon-Entropy = -SUMi->N(probability(pi) * log(probability(pi)))
# Cross entropy can be rearranged where:
# Cross-Entropy = Shannon-Entropy + Kullback-Leibler-Divergence
# CE = -SUMi->N(probability(pi) * log(probability(pi))) + -SUMi->N(probability(pi) * log(probability(qi)/probability(pi)))
# CE = -SUMi->N(probability(pi) * log(probability(pi))) + -SUMi->N(probability(pi) * log(probability(qi)) - (probability(pi) * log(probability(pi)))
# CE = ------------------------------------------------ + -SUMi->N(probability(pi) * log(probability(qi)) - ----------------------------------------
# CE = -SUMi->N(probability(pi) * log(probability(qi))


def kl_divergence(frequencies: np.ndarray, bgd_frequencies: np.ndarray, per_entry: bool = False,
                  mask: np.array = None, axis: int | tuple[int, ...] = None) \
        -> np.ndarray | float:
    """Calculate Kullback–Leibler Divergence entropy between observed and background frequency distribution(s)

    The divergence will be summed across the last axis/dimension of the input array

    Args:
        frequencies: [0.05, 0.001, 0.1, ...] The true distribution
        bgd_frequencies: [0, 0, ...] The model distribution
        per_entry: Whether the result should be returned after summation over the last axis
        mask: A mask to restrict calculations to certain entries
        axis: If the input should be summed over additional axis, which one(s)?
    Returns:
        The additional entropy needed to represent the frequencies as the background frequencies.
            The minimum divergence is 0 when both distributions are identical
    """
    probs1 = bgd_frequencies * np.log(frequencies/bgd_frequencies)
    # if per_residue:
    kl_per_entry = np.sum(np.where(np.isnan(probs1), 0, probs1), axis=-1)
    #     return loss
    if per_entry:
        return -kl_per_entry
    elif mask is None:
        return -np.sum(kl_per_entry, axis=axis)
    else:
        # return -(kl_per_entry * mask) / mask
        return -np.sum(kl_per_entry * mask, axis=axis) / np.sum(mask, axis=axis)


def cross_entropy(frequencies: np.ndarray, bgd_frequencies: np.ndarray, per_entry: bool = False,
                  mask: np.array = None, axis: int | tuple[int, ...] = None) \
        -> np.ndarray | float:
    """Calculate the cross entropy between observed and background frequency distribution(s)

    The entropy will be summed across the last axis/dimension of the input array

    Args:
        frequencies: [0.05, 0.001, 0.1, ...] The true distribution
        bgd_frequencies: [0, 0, ...] The model distribution
        per_entry: Whether the result should be returned after summation over the last axis
        mask: A mask to restrict calculations to certain entries
        axis: If the input should be summed over additional axis, which one(s)?
    Returns:
        The total entropy to represent the frequencies as the background frequencies.
            The minimum entropy is 0 where both distributions are identical
    """
    probs1 = bgd_frequencies * np.log(frequencies)
    # if per_residue:
    ce_per_entry = np.sum(np.where(np.isnan(probs1), 0, probs1), axis=-1)
    #     return loss
    if per_entry:
        return -ce_per_entry
    elif mask is None:
        return -np.sum(ce_per_entry, axis=axis)
    else:
        # return -(ce_per_entry * mask) / mask
        return -np.sum(ce_per_entry * mask, axis=axis) / np.sum(mask, axis=axis)


def js_divergence(frequencies: np.ndarray, bgd_frequencies: np.ndarray, lambda_: float = 0.5) -> float:
    """Calculate Jensen-Shannon Divergence value from observed and background frequencies

    Args:
        frequencies: [0.05, 0.001, 0.1, ...] The true distribution
        bgd_frequencies: [0, 0, ...] The model distribution
        lambda_: Bounded between 0 and 1 indicates weight of the observation versus the background
    Returns:
        Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    r = (lambda_ * frequencies) + ((1 - lambda_) * bgd_frequencies)
    probs1 = frequencies * np.log2(frequencies / r)
    probs2 = bgd_frequencies * np.log2(bgd_frequencies / r)
    return (lambda_ * np.where(np.isnan(probs1), 0, probs1).sum()) \
        + ((1 - lambda_) * np.where(np.isnan(probs2), 0, probs2).sum())


# This is for a multiaxis ndarray
def position_specific_divergence(frequencies: np.ndarray, bgd_frequencies: np.ndarray, lambda_: float = 0.5) -> \
        np.ndarray:
    """Calculate Jensen-Shannon Divergence value from observed and background frequencies

    Args:
        frequencies: [0.05, 0.001, 0.1, ...]
        bgd_frequencies: [0, 0, ...]
        lambda_: Bounded between 0 and 1 indicates weight of the observation versus the background
    Returns:
        An array of divergences bounded between 0 and 1. 1 indicates frequencies are more divergent from background
    """
    r = (lambda_ * frequencies) + ((1 - lambda_) * bgd_frequencies)
    with warnings.catch_warnings():
        # Ignore all warnings related to np.nan
        warnings.simplefilter('ignore')
        probs1 = frequencies * np.log2(frequencies / r)
        probs2 = bgd_frequencies * np.log2(bgd_frequencies / r)
    return (lambda_ * np.where(np.isnan(probs1), 0, probs1).sum(axis=1)) \
        + ((1 - lambda_) * np.where(np.isnan(probs2), 0, probs2).sum(axis=1))

# def js_divergence(frequencies: Sequence[float], bgd_frequencies: Sequence[float], lambda_: float = 0.5) -> \
#         float:
#     """Calculate Jensen-Shannon Divergence value from observed and background frequencies
#
#     Args:
#         frequencies: [0.05, 0.001, 0.1, ...]
#         bgd_frequencies: [0, 0, ...]
#         lambda_: Bounded between 0 and 1 indicates weight of the observation versus the background
#     Returns:
#         Bounded between 0 and 1. 1 is more divergent from background frequencies
#     """
#     sum_prob1, sum_prob2 = 0, 0
#     for frequency, bgd_frequency in zip(frequencies, bgd_frequencies):
#         # bgd_frequency = bgd_frequencies.get(item)
#         try:
#             r = (lambda_ * frequency) + ((1 - lambda_) * bgd_frequency)
#         except TypeError:  # bgd_frequency is None, therefore the frequencies can't be compared. Should error be raised?
#             continue
#         try:
#             with warnings.catch_warnings():
#                 # Cause all warnings to always be ignored
#                 warnings.simplefilter('ignore')
#                 try:
#                     prob2 = bgd_frequency * log(bgd_frequency / r, 2)
#                 except (ValueError, RuntimeWarning):  # math DomainError doesn't raise, instead RunTimeWarn
#                     prob2 = 0
#                 sum_prob2 += prob2
#                 try:
#                     prob1 = frequency * log(frequency / r, 2)
#                 except (ValueError, RuntimeWarning):  # math domain error
#                     continue
#                 sum_prob1 += prob1
#         except ZeroDivisionError:  # r = 0
#             continue
#
#     return lambda_ * sum_prob1 + (1 - lambda_) * sum_prob2


def df_permutation_test(grouped_df: pd.DataFrame, diff_s: pd.Series, group1_size: int = 0, compare: str = 'mean',
                        permutations: int = 1000) -> pd.Series:  # TODO SDUtils
    """Run a permutation test on a dataframe with two categorical groups. Default uses mean to compare significance

    Args:
        grouped_df: The features of interest in samples from two groups of interest. Doesn't need to be sorted
        diff_s: The differences in each feature in the two groups after evaluating the 'compare' stat
        group1_size: Size of the observations in group1
        compare: Choose from any pandas.DataFrame attribute that collapses along a column. Other options might be median
        permutations: The number of permutations to perform
    Returns:
        Contains the p-value(s) of the permutation test using the 'compare' statistic against diff_s
    """
    permut_s_array = []
    df_length = len(grouped_df)
    for i in range(permutations):
        shuffled_df = grouped_df.sample(n=df_length)
        permut_s_array.append(getattr(shuffled_df.iloc[:group1_size, :], compare)().sub(
            getattr(shuffled_df.iloc[group1_size:, :], compare)()))
    # How many times the magnitude of the permuted comparison set is less than the magnitude of the difference set
    # If permuted is less than, returns True, which when taking the mean (or other 'compare'), reflects 1 while False
    # (more than/equal to the difference set) is 0.
    # Essentially, the returned mean is the p-value, which indicates how significant the permutation test results are
    abs_s = diff_s.abs()
    bool_df = pd.DataFrame([permut_s.abs().gt(abs_s) for permut_s in permut_s_array])

    return bool_df.mean()


# def calculate_column_number(num_groups=1, misc=0, sig=0):  # UNUSED, DEPRECIATED
#     total = len(final_metrics) * len(stats_metrics)
#     total += len(significance_columns) * num_groups * len(stats_metrics)
#     total += misc
#     total += sig  # for protocol pair mean similarity measure
#     total += sig * len(significance_columns)  # for protocol pair individual similarity measure
#
#     return total


def filter_df_for_index_by_value(df: pd.DataFrame, metrics: dict[str, list | dict | str | int | float]) \
        -> dict[str, list[Any]]:
    """Retrieve the indices from a DataFrame which have column values passing an indicated operation threshold

    Args:
        df: DataFrame to filter indices on
        metrics: {metric_name: [(operation (Callable), pre_operation (Callable), pre_kwargs (dict), value (Any)),], ...}
            {metric_name: 0.3, ...} OR
            {metric_name: {'direction': 'min', 'value': 0.3}, ...} to specify a sorting direction
    Returns:
        {metric_name: ['0001', '0002', ...], ...}
    """
    filtered_indices = {}
    print_filters = []
    for metric_name, filter_ops in metrics.items():
        if isinstance(filter_ops, list):
            multiple_ops = True if len(filter_ops) > 1 else False
            # Where the metrics = {metric: [(operation, pre_operation, pre_kwargs, value),], ...}
            for idx, filter_op in enumerate(filter_ops, 1):
                operation, pre_operation, pre_kwargs, value = filter_op
                print_filters.append((metric_name, f'{flags.operator_strings[operation]} {value}'))

                try:
                    filtered_df = df[operation(pre_operation(df[metric_name], **pre_kwargs), value)]
                except KeyError:  # metric_name is missing from df
                    logger.error(f"The metric {metric_name} wasn't available in the DataFrame")
                    filtered_df = df
                if multiple_ops:
                    # Add and index as the metric_name could be used a couple of times
                    filter_name = f'{metric_name}({idx})'
                else:
                    filter_name = metric_name
                filtered_indices[filter_name] = filtered_df.index.to_list()
            # Currently below operations aren't necessary because of how index_intersection works
            #  indices = operation1(pre_operation(**kwargs)[metric], value)
            #  AND if more than one argument, only 2 args are possible...
            #  indices = np.logical_and(operation1(pre_operation(**kwargs)[metric], value), operation2(*args))
        else:
            if isinstance(filter_ops, dict):
                specification = filter_ops.get('direction')  # Todo make an ability to use boolean?
                # Todo convert specification options 'greater' '>' 'greater than' to 'max'/'min'
                filter_ops = filter_ops.get('value', 0.)
            else:
                substituted_metric_name = metric_name.translate(utils.remove_digit_table)
                specification = filter_df.loc['direction', substituted_metric_name]

            if specification == 'max':
                filtered_indices[metric_name] = df[df[metric_name] >= filter_ops].index.to_list()
                operator_string = '>='
            elif specification == 'min':
                filtered_indices[metric_name] = df[df[metric_name] <= filter_ops].index.to_list()
                operator_string = '<='
            # Add to the filters
            print_filters.append((metric_name, f'{operator_string} {filter_ops}'))

    # Report the filtering options
    logger.info('Applied filters:\n\t%s' % '\n\t'.join(utils.pretty_format_table(print_filters)))

    return filtered_indices


selection_weight_column = 'selection_weight'


def prioritize_design_indices(df: pd.DataFrame | AnyStr, filters: dict | bool = None, weights: dict | bool = None,
                              protocols: str | list[str] = None, default_weight: str = 'interface_energy', **kwargs) \
        -> pd.DataFrame:
    """Return a filtered/sorted DataFrame (both optional) with indices that pass a set of filters and/or are ranked
    according to a feature importance. Both filter and weight instructions are provided or queried from the user

    Caution: Expects that if DataFrame is provided by filename there is particular formatting, i.e. 3 column
    MultiIndices, 1 index indices. If the DF file varies from this, this function will likely cause errors

    Args:
        df: DataFrame to filter/weight indices
        filters: Whether to remove viable candidates by certain metric values or a mapping of value and filter threshold
            pairs
        weights: Whether to rank the designs by metric values or a mapping of value and weight pairs where the total
            weight will be the sum of all individual weights
        protocols: Whether specific design protocol(s) should be chosen
        default_weight: If there is no weight provided, what is the default metric to sort results
    Keyword Args:
        weight_function: str = 'rank' - The function to use when weighting design indices
    Returns:
        The sorted DataFrame based on the provided filters and weights. DataFrame contains simple Index columns
    """
    # Grab pose info from the DateFrame and drop all classifiers in top two rows.
    if isinstance(df, pd.DataFrame):
        if 3 - df.columns.nlevels > 0:
            df = pd.concat([df], axis=1, keys=[tuple('pose' for _ in range(3 - df.columns.nlevels))])
    else:
        df = pd.read_csv(df, index_col=0, header=[0, 1, 2])
        df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)

    if protocols is not None:
        raise NotImplementedError(f"Can't handle filtering by protocol yet. Fix upstream protocol inclusion in df")
        if isinstance(protocols, str):
            # Add protocol to a list
            protocols = [protocols]

        try:
            protocol_df = df.loc[:, idx_slice[protocols, protocol_column_types, :]]
        except KeyError:
            logger.warning(f"Protocol(s) '{protocols}' weren't found in the set of designs...")
            available_protocols = df.columns.get_level_values(0).unique()
            while True:
                protocols = input(f'What protocol would you like to choose?{describe_string}\n'
                                  f'Available options are: {", ".join(available_protocols)}{input_string}')
                if protocols in available_protocols:
                    protocols = [protocols]  # todo make multiple protocols available for input ^
                    break
                elif protocols in describe:
                    describe_data(df=df)
                else:
                    print(f'Invalid protocol {protocols}. Please choose one of {", ".join(available_protocols)}')
            protocol_df = df.loc[:, idx_slice[protocols, protocol_column_types, :]]
        protocol_df.dropna(how='all', inplace=True, axis=0)  # drop completely empty rows in case of groupby ops
        # ensure 'dock'ing data is present in all protocols
        simple_df = pd.merge(df.loc[:, idx_slice[['pose'], ['dock'], :]], protocol_df, left_index=True, right_index=True)
        logger.info(f'Number of designs after protocol selection: {len(simple_df)}')
    else:
        protocols = ['pose']  # Todo change to :?
        simple_df = df.loc[:, idx_slice[protocols, df.columns.get_level_values(1) != 'std', :]]

    # This is required for a multi-index column where the different protocols are in the top row of the df columns
    simple_df = pd.concat([simple_df.loc[:, idx_slice[prot, :, :]].droplevel(0, axis=1).droplevel(0, axis=1)
                           for prot in protocols])
    simple_df.dropna(how='all', inplace=True, axis=0)
    # simple_df = simple_df.droplevel(0, axis=1).droplevel(0, axis=1)  # simplify headers

    if filters is not None:
        if filters and isinstance(filters, dict):
            # These were passed as parsed values
            pass
        else:  # --filter was provided, but as a boolean-esq dict. Query the user for them
            available_filters = simple_df.columns.to_list()
            filters = query_user_for_metrics(available_filters, df=simple_df, mode='filter', level='design')
        logger.info(f'Number of starting designs: {len(df)}')
        # When df is not ranked by percentage
        # _filters = {metric: {'direction': filter_df.loc['direction', metric], 'value': value}
        #             for metric, value in filters.items()}

        # Filter the DataFrame to include only those values which are le/ge the specified filter
        filtered_indices = filter_df_for_index_by_value(simple_df, filters)  # **_filters)
        # filtered_indices = {metric: filters_with_idx[metric]['idx'] for metric in filters_with_idx}
        logger.info('Number of designs passing filters:\n\t%s' %
                    '\n\t'.join(utils.pretty_format_table([(metric, '=', len(indices))
                                                           for metric, indices in filtered_indices.items()])))
        # logger.info('Number of designs passing filters:\n\t%s'
        #             % '\n\t'.join(f'{len(indices):6d} - {metric}' for metric, indices in filtered_indices.items()))
        final_indices = index_intersection(filtered_indices.values())
        number_final_indices = len(final_indices)
        if number_final_indices == 0:
            raise DesignError('There are no poses left after filtering. Try choosing less stringent values')
        logger.info(f'Number of designs passing all filters: {number_final_indices}')
        simple_df = simple_df.loc[final_indices, :]

    # {column: {'direction': min_, 'value': 0.3, 'idx_slice': ['0001', '0002', ...]}, ...}
    # if weight is not None or default_weight in simple_df.columns:
    if weights:
        if isinstance(weights, dict):
            # These were passed as parsed values
            pass
        else:  # --weight was provided, but as a boolean-esq dict. Query the user for them
            available_metrics = simple_df.columns.to_list()
            weights = query_user_for_metrics(available_metrics, df=simple_df, mode='weight', level='design')
    elif default_weight in simple_df.columns:
        weights = None
    else:
        raise KeyError(f"No 'weight' provided and couldn't find the metric key {default_weight} in the DataFrame. "
                       f"Available metric keys: {simple_df.columns.tolist()}")
    ranking_s = pareto_optimize_trajectories(simple_df, weights=weights, default_weight=default_weight, **kwargs)
    # Using the sorted indices of the ranking_s, rename, then join the existing df indices to it
    # This maintains ranking order
    final_df = ranking_s.rename(selection_weight_column).to_frame().join(simple_df)

    return final_df


describe_string = '\nTo see a description of the data, enter "describe"\n'
describe = ['describe', 'desc', 'DESCRIBE', 'DESC', 'Describe', 'Desc']


def describe_data(df: pd.DataFrame = None) -> None:
    """Describe the DataFrame to STDOUT"""
    print('The available metrics are located in the top row(s) of your DataFrame. Enter your selected metrics as a '
          'comma separated input. To see descriptions for only certain metrics, enter them here. '
          'Otherwise, hit "Enter"')
    metrics_input = input(input_string)
    chosen_metrics = set(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                            repeat(' '), repeat('_'))))\
        .difference({''})  # Remove "Enter" input if that was provided

    if not chosen_metrics:
        columns_of_interest = slice(None)
    else:
        columns_of_interest = [idx for idx, column in enumerate(df.columns.get_level_values(-1).to_list())
                               if column in chosen_metrics]
    # Format rows/columns for data display, then revert
    max_columns, min_columns = pd.get_option('display.max_columns'), pd.get_option('display.max_rows')
    pd.set_option('display.max_columns', None), pd.set_option('display.max_rows', None)
    print(df.iloc[:, columns_of_interest].describe())
    pd.set_option('display.max_columns', max_columns), pd.set_option('display.max_rows', min_columns)


@utils.handle_errors(errors=(KeyboardInterrupt,))
def query_user_for_metrics(available_metrics: Iterable[str], df: pd.DataFrame = None, mode: str = None,
                           level: str = None) -> dict[str, float]:
    """Ask the user for the desired metrics to select indices from a dataframe

    Args:
        available_metrics: The columns available in the DataFrame to select indices by
        df: A DataFrame from which to use metrics (provided as columns)
        mode: The mode in which to query and format metrics information. Either 'filter' or weight'
        level: The hierarchy of selection to use. Could be one of 'poses', 'designs', or 'sequences'
    Returns:
        The mapping of metric name to value
    """
    try:
        direction = dict(max='higher', min='lower')
        instructions = \
            {'filter': '\nFor each metric, choose values based on supported literature or design goals to eliminate '
                       "designs that are certain to fail or have sub-optimal features. Ensure your cutoffs aren't too "
                       'exclusive. If you end up with no designs, try relaxing your filter values.',
             'weight':
                 '\nFor each metric, choose a percentage signifying the metrics contribution to the total selection '
                 'weight. The weight will be used as a linear combination of all weights according to each designs rank'
                 ' within the specified metric category. For instance, typically the total weight should equal 1. When '
                 'choosing 5 metrics, you can assign an equal weight to each (specify 0.2 for each) or you can weight '
                 'several more strongly (0.3, 0.3, 0.2, 0.1, 0.1). When ranking occurs, for each selected metric the '
                 'metric will be sorted and designs in the top percentile will be given their percentage of the full '
                 'weight. Top percentile is defined as the most advantageous score, so the top percentile of energy is '
                 'lowest, while for hydrogen bonds it would be the most.'}

        print('\n%s' % header_string % f'Select {level} {mode} Metrics')
        print(f'The provided dataframe will be used to select {level}s based on the measured metrics from each pose. '
              f'To "{mode}" designs, which metrics would you like to utilize?'
              f'{"" if df is None else describe_string}')

        print('The available metrics are located in the top row(s) of your DataFrame. Enter your selected metrics as a '
              'comma separated input or alternatively, you can check out the available metrics by entering "metrics".'
              '\nEx: "shape_complementarity, contact_count, etc."')
        metrics_input = input(input_string)
        chosen_metrics = set(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                                repeat(' '), repeat('_'))))
        available_metrics = sorted(available_metrics)
        while True:  # unsupported_metrics or 'metrics' in chosen_metrics:
            unsupported_metrics = chosen_metrics.difference(available_metrics)
            if 'metrics' in chosen_metrics:
                print(f'You indicated "metrics". Here are available metrics:\n{", ".join(available_metrics)}\n')
                metrics_input = input(input_string)
            elif chosen_metrics.intersection(describe):
                describe_data(df=df) if df is not None else print("Can't describe data without providing a DataFrame")
                # df.describe() if df is not None else print('Can\'t describe data without providing a DataFrame...')
                metrics_input = input(input_string)
            elif unsupported_metrics:
                # TODO catch value error in dict comprehension upon string input
                metrics_input = input(f'Metric{"s" if len(unsupported_metrics) > 1 else ""} '
                                      f'"{", ".join(unsupported_metrics)}" not found in the DataFrame!'
                                      '\nIs your spelling correct? Have you used the correct underscores? '
                                      f'Please input these metrics again. Specify "metrics" to view available metrics'
                                      f'{input_string}')
            elif len(chosen_metrics) > 0:
                # We have no errors and there are metrics
                break
            else:
                input_flag = flags.format_args(flags.filter_args) if mode == "filter" \
                    else flags.format_args(flags.weight_args)
                print("Metrics weren't provided... If this is what you want, run this module without the "
                      f'{input_flag} flag')
                if verify_choice():
                    break
            fixed_metrics = list(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                                    repeat(' '), repeat('_'))))
            chosen_metrics = chosen_metrics.difference(unsupported_metrics).union(fixed_metrics)
            # unsupported_metrics = set(chosen_metrics).difference(available_metrics)

        print(instructions[mode])
        while True:  # not correct:  # correct = False
            print("" if df is None else describe_string)
            metric_values = {}
            for metric in chosen_metrics:
                # Modify the provided metric of digits to get its configuration info
                substituted_metric = metric.translate(utils.remove_digit_table)
                while True:
                    # Todo make ability to use boolean descriptions
                    # Todo make ability to specify direction
                    value = input(f'For "{metric}" what value should be used for {level} {mode}ing? %s{input_string}'
                                  % ('Designs with metrics %s than this value will be included' %
                                     direction[filter_df.loc['direction', substituted_metric]].upper()
                                     if mode == "filter" else ""))
                    if value in describe:
                        describe_data(df=df) if df is not None \
                            else print("Can't describe data without providing a DataFrame...")
                    elif validate_type(value, dtype=float):
                        metric_values[metric] = float(value)
                        break

            # metric_values = {metric: float(input('For "%s" what value should be used for %s %sing?%s%s'
            #                                      % (metric, level, mode,
            #                                         ' Designs with metrics %s than this value will be included'
            #                                         % direction[filter_df.loc['direction', metric]].upper()
            #                                         if mode == 'filter' else '', input_string)))
            #                  for metric in chosen_metrics}
            if metric_values:
                print('You selected:\n\t%s' % '\n\t'.join(utils.pretty_format_table(metric_values.items())))
            else:
                # print('No metrics were provided, skipping value input')
                # metric_values = None
                break

            if verify_choice():
                break
    except KeyboardInterrupt:
        exit('\nSelection was ended by Ctrl-C!')

    return metric_values


def pareto_optimize_trajectories(df: pd.DataFrame, weights: dict[str, float] = None,
                                 function: config.weight_functions_literal = 'rank',
                                 default_weight: str = 'interface_energy', **kwargs) -> pd.Series:
    """From a provided DataFrame with individual design trajectories, select trajectories based on provided metric and
    weighting parameters

    Args:
        df: The designs x metrics DataFrame (single index metrics column) to select trajectories from
        weights: {'metric': value, ...}. If not provided, sorts by default_sort
        function: The function to use for weighting. Either 'rank' or 'normalize' is possible
        default_weight: The metric to weight the dataframe by default if no weights are provided
    Returns:
        A sorted pandas.Series with the best indices first in the Series.index, and the resulting optimization values
        in the corresponding value.
    """
    if weights:  # Could be None or empty dict
        # weights = {metric: dict(direction=filter_df.loc['direction', metric], value=value)
        #            for metric, value in weights.items()}
        coefficients = {}
        print_weights = []
        for metric_name, weight_ops in weights.items():
            # Modify the provided metric of digits to get its configuration info
            substituted_metric = metric_name.translate(utils.remove_digit_table)
            direction = filter_df.loc['direction', substituted_metric]
            if isinstance(weight_ops, list):
                # Where the metrics = {metric: [(operation, pre_operation, pre_kwargs, value),], ...}
                # Currently, can only have one weight per metric...
                for idx, weight_op in enumerate(weight_ops):
                    operation, pre_operation, pre_kwargs, value = weight_op
                    coefficients[metric_name] = dict(direction=direction, value=value)
                    print_weights.append((metric_name, f'= {value}'))
            else:  # weight_ops is just the value
                coefficients[metric_name] = dict(direction=direction, value=weight_ops)
                print_weights.append((metric_name, f'= {weight_ops}'))

        metric_df = {}
        if function == 'rank':
            # This puts small and negative value (when min is chosen) with higher rank
            sort_direction = dict(max=True, min=False)  # max - ascending=True, min - ascending=False

            for metric_name, parameters in coefficients.items():
                direction = parameters['direction']
                try:
                    metric_series = \
                        df[metric_name].rank(ascending=sort_direction[direction], method=direction, pct=True) \
                        * parameters['value']
                except KeyError:  # metric_name is missing from df
                    logger.error(f"{pareto_optimize_trajectories.__name__}: The metric {metric_name} wasn't available "
                                 "for weighting in the given DataFrame")
                    continue
                metric_df[metric_name] = metric_series
            # df = pd.concat({metric: df[metric].rank(ascending=sort_direction[parameters['direction']],
            #                                         method=parameters['direction'], pct=True) * parameters['value']
            #                 for metric, parameters in weights.items()}, axis=1)
        elif function == 'normalize':  # Get the MinMax normalization (df - df.min()) / (df.max() - df.min())
            for metric_name, parameters in coefficients.items():
                metric_s = df[metric_name]
                if parameters['direction'] == 'max':
                    metric_min = metric_s.min()
                    metric_max = metric_s.max()
                else:  # parameters['direction'] == 'min':
                    metric_min = metric_s.max()
                    metric_max = metric_s.min()
                metric_df[metric_name] = \
                    ((metric_s - metric_min) / (metric_max - metric_min)) * parameters['value']
        else:
            raise ValueError(f"The value {function} isn't a viable choice for metric weighting 'function'")

        if metric_df:
            logger.info('Applied weights:\n\t%s' % '\n\t'.join(utils.pretty_format_table(print_weights)))
            weighted_df = pd.concat(metric_df, axis=1)
            return weighted_df.sum(axis=1).sort_values(ascending=False)

    if default_weight in df.columns:
        # For sort_values(), this sorts the right direction, while for rank() it sorts incorrectly
        sort_direction = dict(max=False, min=True)  # max - ascending=False, min - ascending=True

        # Just sort by the default
        direction = filter_df.loc['direction', default_weight]
        # return df.sort_values(default_sort, ascending=sort_direction[direction])
        return df[default_weight].sort_values(ascending=sort_direction[direction])
    else:
        raise KeyError(f"There wasn't a metric named '{default_weight}' which was specified as the default")


def window_function(data: Sequence[int | float], windows: Iterable[int] = None, lower: int = None,
                    upper: int = None) -> np.ndarray:
    """Perform windowing operations on a sequence of data and return the result of the calculation. Window lengths can
    be specified by passing the windows to perform calculation on as an Iterable or by a range of window lengths

    Args:
        data: The sequence of numeric data to perform calculations
        windows: An iterable of window lengths to use. If a single, pass as the Iterable
        lower: The lower range of the window to operate on. "window" is inclusive of this value
        upper: The upper range of the window to operate on. "window" is inclusive of this value
    Returns:
        The (number of windows, length of data) array of values with each requested window along axis=0
            and the particular value of the windowed data along axis=1
    """
    array_length = len(data)
    if windows is None:
        if lower is not None and upper is not None:
            windows = list(range(lower, upper + 1))  # +1 makes inclusive in range
        else:
            raise ValueError(f'{window_function.__name__}:'
                             f' Must provide either window, or lower and upper')

    # Make an array with axis=0 equal to number of windows used, axis=1 equal to length of values
    # range_size = len(windows)
    # data_template = [0 for _ in range(array_length)]
    window_array = np.zeros((len(windows), array_length))
    for array_idx, window_size in enumerate(windows):  # Make the divisor a float
        half_window = math.floor(window_size / 2)  # how far on each side should the window extend
        # # Calculate score accordingly, with cases for N- and C-terminal windows
        # for data_idx in range(half_window):  # N-terminus windows
        #     # add 1 as high slice not inclusive
        #     window_array[array_idx, data_idx] = sequence_array[:data_idx + half_window + 1].sum() / window_size
        # for data_idx in range(half_window, array_length-half_window):  # continuous length windows
        #     # add 1 as high slice not inclusive
        #     window_array[array_idx, data_idx] = \
        #         sequence_array[data_idx - half_window: data_idx + half_window+1].sum() / window_size
        # for data_idx in range(array_length-half_window, array_length):  # C-terminus windows
        #     # No add 1 as low slice inclusive
        #     window_array[array_idx, data_idx] = sequence_array[data_idx - half_window:].sum() / window_size
        #
        # # check if the range is even, then subtract 1/2 of the value of trailing and leading window values
        # if window_size % 2 == 0.:
        #     # subtract_half_leading_residue = sequence_array[half_window:] * 0.5 / window_size
        #     window_array[array_idx, :array_length - half_window] -= \
        #         sequence_array[half_window:] * 0.5 / window_size
        #     # subtract_half_trailing_residue = sequence_array[:array_length - half_window] * 0.5 / window_size
        #     window_array[array_idx, half_window:] -= \
        #         sequence_array[:array_length - half_window] * 0.5 / window_size

        # Calculate score accordingly, with cases for N- and C-terminal windows
        # array_length_range = range(array_length)
        # # Make a "zeros" list
        # data_window = [0 for _ in range(array_length)]
        # window_data = copy(data_template)
        # This would be the method if the slices need to be taken with respect to the c-term
        # for end_idx, start_idx in enumerate(range(array_length - window_size), window_size):
        # There is no off by one error if we slice lower or higher than list so include both termini
        # for end_idx, start_idx in enumerate(range(array_length), window_size):
        #     idx_sum = sum(data[start_idx:end_idx])
        #     # for window_position in range(start_idx, end_idx + 1):
        #     # # for window_position in range(data_idx - half_window, data_idx + half_window + 1):
        #     #     idx_sum += sum(data[start_idx:end_idx])
        #     window_data[data_idx] = idx_sum

        # Calculate each score given the window. Accounts for window cases with N- and C-termini
        # There is no off by one error if we slice lower or higher than list so include both termini
        window_data = [sum(data[start_idx:end_idx])
                       for end_idx, start_idx in enumerate(range(-array_length - half_window, -half_window),
                                                           half_window + 1)]

        # # Old python list method
        # for data_idx in array_length_range:
        #     idx_sum = 0
        #     if data_idx < half_window:  # N-terminus
        #         for window_position in range(data_idx + half_window + 1):
        #             idx_sum += data[window_position]
        #     elif data_idx + half_window >= array_length:  # C-terminus
        #         for window_position in range(data_idx - half_window, array_length):
        #             idx_sum += data[window_position]
        #     else:
        #         for window_position in range(data_idx - half_window, data_idx + half_window + 1):
        #             idx_sum += data[window_position]
        #
        #     # Set each idx_sum to the idx in data_window
        #     data_window[data_idx] = idx_sum
        # Handle data_window incorporation into numpy array
        window_array[array_idx] = window_data
        window_array[array_idx] /= float(window_size)

        # Account for windows that have even ranges
        if window_size % 2 == 0.:  # The range is even
            # Calculate a modifier to subtract from each of the data values given the original value and the window size
            even_modifier = .5 / window_size
            even_modified_data = [value * even_modifier for value in data]
            # subtract_half_leading_residue = sequence_array[half_window:] * 0.5 / window_size
            window_array[array_idx, :-half_window] -= even_modified_data[half_window:]
            # subtract_half_trailing_residue = sequence_array[:array_length - half_window] * 0.5 / window_size
            window_array[array_idx, half_window:] -= even_modified_data[:-half_window]

    return window_array


hydrophobicity_scale = \
    {'expanded': {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1, 'M': 1, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 1, 'Y': 0, 'B': 0, 'J': 0, 'O': 0, 'U': 0,
                  'X': 0, 'Z': 0},
     'standard': {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1, 'M': 0, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 0, 'Y': 0, 'B': 0, 'J': 0, 'O': 0, 'U': 0,
                  'X': 0, 'Z': 0}}
hydrophobicity_scale_literal = Literal['expanded', 'standard']


def hydrophobic_collapse_index(seq: Sequence[str | int] | np.ndarry,
                               hydrophobicity: hydrophobicity_scale_literal = 'standard',
                               custom: dict[sequence.protein_letters_literal, int | float] = None,
                               alphabet_type: sequence.alphabet_types_literal = None,
                               lower_window: int = 3, upper_window: int = 9, **kwargs) -> np.ndarray:
    """Calculate hydrophobic collapse index for sequence(s) of interest and return an HCI array

    Args:
        seq: The sequence to measure. Can be a character based sequence (or array of sequences with shape
            (sequences, residues)), an integer based sequence, or a sequence profile like array (residues, alphabet)
            where each character in the alphabet contains a typical distribution of amino acid observations
        hydrophobicity: The hydrophobicity scale to consider. Either 'standard' (FILV), 'expanded' (FMILVW),
            or provide one with the keyword argument, "custom"
        custom: A user defined mapping of amino acid type, hydrophobicity value pairs
        alphabet_type: The amino acid alphabet if seq consists of integer characters
        lower_window: The smallest window used to measure
        upper_window: The largest window used to measure
    Returns:
        1D array with the hydrophobic collapse index at every position on the input sequence(s)
    """
    if custom is None:
        hydrophobicity_values = hydrophobicity_scale.get(hydrophobicity)
        # hydrophobicity == 'background':  # Todo
        if not hydrophobicity_values:
            raise ValueError(f'The hydrophobicity "{hydrophobicity}" table is not available. Add it if you think it '
                             f'should be')
    else:
        hydrophobicity_values = custom

    def solve_alphabet() -> sequence.alphabets_literal:
        if alphabet_type is None:
            raise ValueError(
                f'{hydrophobic_collapse_index.__name__}: Must pass keyword "alphabet_type" when calculating '
                f'using integer sequence values')
        else:
            alphabet_ = sequence.alphabet_type_to_alphabet.get(alphabet_type)
            if alphabet_ is None:
                if sequence.alphabet_to_alphabet_type.get(alphabet_type):
                    alphabet_ = alphabet_type
                else:
                    raise ValueError(
                        f"{hydrophobic_collapse_index.__name__}: alphabet_type '{alphabet_type}' isn't a viable "
                        f'alphabet_type. Choose from {", ".join(sequence.alphabet_types)} or pass an alphabet')

            return alphabet_

    if isinstance(seq[0], int):  # This is an integer sequence. An alphabet is required
        alphabet = solve_alphabet()
        values = [hydrophobicity_values[aa] for aa in alphabet]
        sequence_array = [values[aa_int] for aa_int in seq]
        # raise ValueError(f"sequence argument with type {type(sequence).__name__} isn't supported")
    elif isinstance(seq[0], str):  # This is a string array # if isinstance(sequence[0], str):
        sequence_array = [hydrophobicity_values.get(aa, 0) for aa in seq]
        # raise ValueError(f"sequence argument with type {type(sequence).__name__} isn't supported")
    elif isinstance(seq, (torch.Tensor, np.ndarray)):  # This is an integer sequence. An alphabet is required
        if seq.dtype in utils.np_torch_int_float_types:
            alphabet = solve_alphabet()
            # torch.Tensor and np.ndarray can multiply by np.ndarray
            values = np.array([hydrophobicity_values[aa] for aa in alphabet])
            if seq.ndim == 2:
                # print('HCI debug')
                # print('array.shape', seq.shape, 'values.shape', values.shape)
                # The array must have shape (number_of_residues, alphabet_length)
                sequence_array = seq * values
                # Ensure each position is a combination of the values for each amino acid
                sequence_array = sequence_array.sum(axis=-1)
                # print('sequence_array', sequence_array)
            else:
                raise ValueError(f"Can't process a {seq.ndim}-dimensional array yet")
        else:  # We assume it is a sequence array with bytes?
            # The array must have shape (number_of_residues, alphabet_length)
            sequence_array = seq * np.vectorize(hydrophobicity_values.__getitem__)(seq)
            # Ensure each position is a combination of the values for each amino acid in the array
            sequence_array = sequence_array.mean(axis=-2)
        # elif isinstance(sequence, Sequence):
        #     sequence_array = [hydrophobicity_values.get(aa, 0) for aa in sequence]
    else:
        raise ValueError(f'The provided sequence must comprise the canonical amino acid string characters or '
                         f'integer values corresponding to numerical amino acid conversions. '
                         f'Got type={type(seq[0]).__name__} instead')

    window_array = window_function(sequence_array, lower=lower_window, upper=upper_window)

    return window_array.mean(axis=0)


def index_intersection(index_groups: Iterable[Iterable[Any]]) -> list[Any]:
    """Perform AND logic on objects in multiple containers of objects, where all objects must be present to be included

    Args:
        index_groups: Groups of indices
    Returns:
        The union of all provided indices
    """
    final_indices = set()
    # Find all set union. This grabs every possible index
    for indices in index_groups:
        final_indices = final_indices.union(indices)
    # Find all set intersection. This narrows down to those present only in all
    for indices in index_groups:
        final_indices = final_indices.intersection(indices)

    return list(final_indices)


@jit(nopython=True)  # , cache=True)
def z_score(sample: float | np.ndarray, mean: float | np.ndarray, stdev: float | np.ndarray) -> float | np.ndarray:
    """From sample(s), calculate the positional z-score, i.e. z-score = (sample - mean) / stdev

    Args:
        sample: An array with the sample at every position
        mean: An array with the mean at every position
        stdev: An array with the standard deviation at every position
    Returns:
        The z-score of every sample
    """
    # try:
    return (sample-mean) / stdev
    # except ZeroDivisionError:
    #     logger.error('The passed standard deviation (stdev) was 0! z-score calculation failed')
    #     return 0.
