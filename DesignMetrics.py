import math
import operator
from copy import copy, deepcopy
from json import loads

import numpy as np
import pandas as pd

from Structure import gxg_sasa
from SymDesignUtils import start_log, DesignError, index_intersection, handle_errors, \
    pretty_format_table, digit_translate_table
from Query.PDB import header_string, input_string, confirmation_string, bool_d, invalid_string

# Globals
logger = start_log(name=__name__)
index_offset = 1
groups = 'protocol'
master_metrics = {'average_fragment_z_score':
                      {'description': 'The average fragment z-value used in docking/design',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'buns_heavy_total':
                      {'description': 'Buried unsaturated H-bonding heavy atoms in the design',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'buns_hpol_total':
                      {'description': 'Buried unsaturated H-bonding polarized hydrogen atoms in the design',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'buns_total':
                      {'description': 'Total buried unsaturated H-bonds in the design',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'buns_per_ang':
                      {'description': 'Buried Unsaturated Hbonds per Angstrom^2 of interface',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'component_1_symmetry':
                      {'description': 'The symmetry group of component 1',
                       'direction': 'min', 'function': 'equals', 'filter': True},
                  'component_1_name':
                      {'description': 'component 1 PDB_ID', 'direction': None, 'function': None, 'filter': False},
                  'component_1_number_of_residues':
                      {'description': 'The number of residues in the monomer of component 1',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'component_1_max_radius':
                      {'description': 'The maximum distance that component 1 reaches away from the center of mass',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'component_1_n_terminal_helix':
                      {'description': 'Whether the n-terminus has an alpha helix',
                       'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  'component_1_c_terminal_helix':
                      {'description': 'Whether the c-terminus has an alpha helix',
                       'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  'component_1_n_terminal_orientation':
                      {'description': 'The direction the n-terminus is oriented from the symmetry group center of mass.'
                                      ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
                  'component_1_c_terminal_orientation':
                      {'description': 'The direction the c-terminus is oriented from the symmetry group center of mass.'
                                      ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
                  'component_2_symmetry':
                      {'description': 'The symmetry group of component 2',
                       'direction': 'min', 'function': 'equals', 'filter': True},
                  'component_2_name':
                      {'description': 'component 2 PDB_ID', 'direction': None, 'function': None, 'filter': False},
                  'component_2_number_of_residues':
                      {'description': 'The number of residues in the monomer of component 2',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'component_2_max_radius':
                      {'description': 'The maximum distance that component 2 reaches away from the center of mass',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'component_2_n_terminal_helix':
                      {'description': 'Whether the n-terminus has an alpha helix',
                       'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  'component_2_c_terminal_helix':
                      {'description': 'Whether the c-terminus has an alpha helix',
                       'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  'component_2_n_terminal_orientation':
                      {'description': 'The direction the n-terminus is oriented from the symmetry group center of mass.'
                                      ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
                  'component_2_c_terminal_orientation':
                      {'description': 'The direction the c-terminus is oriented from the symmetry group center of mass.'
                                      ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
                  'contact_count':
                      {'description': 'Number of carbon-carbon contacts across interface',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'core':
                      {'description': 'The number of \'core\' residues as classified by E. Levy 2010',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'coordinate_constraint':
                      {'description': 'Total weight of coordinate constraints to keep design from moving in cartesian '
                                      'space', 'direction': 'min', 'function': 'normalize', 'filter': True},
                  'design_dimension':
                      {'description': 'The underlying dimension of the design. 0 - point, 2 - layer, 3 - space group',
                                      'direction': 'min', 'function': 'normalize', 'filter': True},
                  'divergence_design_per_residue':
                      {'description': 'The Jensen-Shannon divergence of interface residues from the position specific '
                                      'design profile values. Includes fragment & evolution if both are True, otherwise'
                                      ' only includes those specified for use in design.',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'divergence_fragment_per_residue':
                      {'description': 'The Jensen-Shannon divergence of interface residues from the position specific '
                                      'fragment profile', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'divergence_evolution_per_residue':
                      {'description': 'The Jensen-Shannon divergence of interface residues from the position specific '
                                      'evolutionary profile', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'divergence_interface_per_residue':
                      {'description': 'The Jensen-Shannon divergence of interface residues from the typical interface '
                                      'background', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'favor_residue_energy':
                      {'description': 'Total weight of sequence constraints used to favor certain amino acids in design'
                                      '. Only protocols with a favored profile have values',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'interaction_energy_complex':
                      {'description': 'The two-body (residue-pair) energy of the complexed interface. No solvation '
                                      'energies', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_area_hydrophobic':
                      {'description': 'Total hydrophobic interface buried surface area',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_area_polar':
                      {'description': 'Total polar interface buried surface area',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_area_total':
                      {'description': 'Total interface buried surface area',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_b_factor_per_residue':
                      {'description': 'The average B-factor from each atom, from each interface residue',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_bound_activation_energy':
                      {'description': 'Energy required for the unbound interface to adopt the conformation in the '
                                      'complexed state', 'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_buried_hbonds':
                      {'description': 'Total buried unsaturated H-bonds in the design',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_composition_similarity':
                      {'description': 'The similarity to the expected interface composition given interface buried '
                                      'surface area. 1 is similar to natural interfaces, 0 is dissimilar',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_connectivity_1':
                      {'description': 'How embedded is interface1 in the rest of the protein?',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'interface_connectivity_2':
                      {'description': 'How embedded is interface2 in the rest of the protein?',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'int_energy_density':
                      {'description': 'Energy in the bound complex per Angstrom^2 of interface area',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_energy':
                      {'description': 'DeltaG of the complexed and unbound (repacked) interfaces',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_energy_complex':
                      {'description': 'Total interface residue energy summed in the complexed state',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_energy_density':
                      {'description': 'Interface energy per interface area^2. How much energy is achieved within the '
                                      'given space?', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_energy_unbound':
                      {'description': 'Total interface residue energy summed in the unbound state',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_energy_1_unbound':
                      {'description': 'Sum of interface1 residue energy in the unbound state',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_energy_2_unbound':
                      {'description': 'Sum of interface2 residue energy in the unbound state',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_separation':
                      {'description': 'Median distance between all atom points on each side of the interface',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'multiple_fragment_ratio':
                      {'description': 'The extent to which fragment observations are connected in the interface. Higher'
                                      ' ratio means multiple fragment observations per residue',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'number_hbonds':
                      {'description': 'The number of residues making H-bonds in the total interface. Residues may make '
                                      'more than one H-bond', 'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score':
                      {'description': 'Sum of total fragment containing residue match scores (1 / 1 + Z-score^2) '
                                      'weighted by their ranked match score. Maximum of 2/residue',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score_center':
                      {'description': 'nanohedra_score for the central fragment residues only',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score_normalized':
                      {'description': 'The Nanohedra Score normalized by number of fragment residues',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score_center_normalized':
                      {'description': 'The central Nanohedra Score normalized by number of central fragment residues',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'number_fragment_residues_total':
                      {'description': 'The number of residues in the interface with fragment observationsfound',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'number_fragment_residues_center':
                      {'description': 'The number of interface residues that belong to a central fragment residue',
                       'direction': 'max', 'function': 'rank', 'filter': None},
                  'observations':
                      {'description': 'Number of unique design trajectories contributing to statistics',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'observed_design':
                      {'description': 'Percent of observed residues in combined profile. 1 is 100%',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'observed_evolution':
                      {'description': 'Percent of observed residues in evolutionary profile. 1 is 100%',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'observed_fragment':
                      {'description': 'Percent of observed residues in the fragment profile. 1 is 100%',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'observed_interface':
                      {'description': 'Percent of observed residues in fragment profile. 1 is 100%',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'percent_core':
                      {'description': 'The percentage of residues which are \'core\' according to Levy, E. 2010',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_fragment':
                      {'description': 'Percent of residues with fragment data out of total residues',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_fragment_coil':
                      {'description': 'The percentage of fragments represented from coiled SS elements',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_fragment_helix':
                      {'description': 'The percentage of fragments represented from an a-helix SS elements',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_fragment_strand':
                      {'description': 'The percentage of fragments represented from a b-strand SS elements',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_interface_area_hydrophobic':
                      {'description': 'The percent of interface area which is occupied by hydrophobic atoms',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'percent_interface_area_polar':
                      {'description': 'The percent of interface area which is occupied by polar atoms',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_residues_fragment_center':
                      {'description': 'The percentage of residues which are central fragment observations',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_residues_fragment_total':
                      {'description': 'The percentage of residues which are represented by fragment observations',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'percent_rim':
                      {'description': 'The percentage of residues which are \'rim\' according to Levy, E. 2010',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'percent_support':
                      {'description': 'The percentage of residues which are \'support\' according to Levy, E. 2010',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  groups:
                      {'description': 'Protocols utilized to search sequence space given fragment and/or evolutionary '
                                      'constraint information', 'direction': None, 'function': None, 'filter': False},
                  'protocol_energy_distance_sum':
                      {'description': 'The distance between the average linearly embedded per residue energy '
                                      'co-variation between specified protocols. Larger = greater distance. A small '
                                      'distance indicates that different protocols arrived at the same per residue '
                                      'energy conclusions despite different pools of amino acids specified for sampling'
                          , 'direction': 'min', 'function': 'rank', 'filter': True},
                  'protocol_similarity_sum':
                      {'description': 'The statistical similarity between all sampled protocols. Larger is more similar'
                                      ', indicating that different protocols have interface statistics that are similar'
                                      ' despite different pools of amino acids specified for sampling',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'protocol_sequence_distance_sum':
                      {'description': 'The distance between the average linearly embedded sequence differences between '
                                      'specified protocols. Larger = greater distance. A small distance indicates that '
                                      'different protocols arrived at the same per residue energy conclusions despite '
                                      'different pools of amino acids specified for sampling',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'rosetta_reference_energy':
                      {'description': 'Rosetta Energy Term - A metric for the unfolded energy of the protein along with'
                                      ' sequence fitting corrections',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'rim':
                      {'description': 'The number of \'rim\' residues as classified by E. Levy 2010',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'rmsd':
                      {'description': 'Root Mean Square Deviation of all CA atoms between the refined (relaxed) and '
                                      'designed states', 'direction': 'min', 'function': 'normalize', 'filter': True},
                  'shape_complementarity':
                      {'description': 'Measure of fit between two surfaces from Lawrence and Colman 1993',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'solvation_energy':  # free_energy of desolvation is positive for bound interfaces. unbound - bound
                      {'description': 'The free energy resulting from hydration of the separated interface surfaces. '
                                      'Positive values indicate poorly soluble surfaces',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'solvation_energy_bound':
                      {'description': 'The desolvation free energy of the separated interface surfaces. Positive values'
                                      ' indicate energy is required to desolvate',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'solvation_energy_complex':
                      {'description': 'The desolvation free energy of the complexed interface. Positive values'
                                      ' indicate energy is required to desolvate',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'solvation_energy_unbound':
                      {'description': 'The desolvation free energy of the separated, repacked, interface surfaces. '
                                      'Positive values indicate energy is required to desolvate',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'support':
                      {'description': 'The number of \'support\' residues as classified by E. Levy 2010',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'symmetry':
                      {'description': 'The specific symmetry type used design (point (0), layer (2), lattice(3))',
                       'direction': None, 'function': None, 'filter': False},
                  # 'fragment_z_score_total':
                  #     {'description': 'The sum of all fragments z-values',
                  #      'direction': None, 'function': None, 'filter': None},
                  'number_of_fragments':
                      {'description': 'The number of fragments found in the pose interface',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'number_of_mutations':
                      {'description': 'The number of mutations made to the pose (ie. wild-type residue to any other '
                                      'amino acid)', 'direction': 'min', 'function': 'normalize', 'filter': True},
                  'total_interface_residues':
                      {'description': 'The total number of interface residues found in the pose (residue CB within 8A)',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'total_non_fragment_interface_residues':
                      {'description': 'The number of interface residues that are missing central fragment observations',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'REU':
                      {'description': 'Rosetta Energy Units. Always 0. We can disregard',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'time':
                      {'description': 'Time for the protocol to complete',
                       'direction': None, 'function': None, 'filter': None},
                  'hbonds_res_selection_unbound':
                      {'description': 'The specific h-bonds present in the bound pose',
                       'direction': None, 'function': None, 'filter': None},
                  'hbonds_res_selection_1_unbound':
                      {'description': 'The specific h-bonds present in the unbound interface1',
                       'direction': None, 'function': None, 'filter': None},
                  'hbonds_res_selection_2_unbound':
                      {'description': 'The specific h-bonds present in the unbound interface2',
                       'direction': None, 'function': None, 'filter': None},
                  'dslf_fa13':
                      {'description': 'Rosetta Energy Term - disulfide bonding',
                       'direction': None, 'function': None, 'filter': None},
                  'fa_atr':
                      {'description': 'Rosetta Energy Term - lennard jones full atom atractive forces',
                       'direction': None, 'function': None, 'filter': None},
                  'fa_dun':
                      {'description': 'Rosetta Energy Term - Dunbrack rotamer library statistical probability',
                       'direction': None, 'function': None, 'filter': None},
                  'fa_elec':
                      {'description': 'Rosetta Energy Term - full atom electrostatic forces',
                       'direction': None, 'function': None, 'filter': None},
                  'fa_intra_rep':
                      {'description': 'Rosetta Energy Term - lennard jones full atom intra-residue repulsive forces',
                       'direction': None, 'function': None, 'filter': None},
                  'fa_intra_sol_xover4':
                      {'description': 'Rosetta Energy Term - full atom intra-residue solvent forces',
                       'direction': None, 'function': None, 'filter': None},
                  'fa_rep':
                      {'description': 'Rosetta Energy Term - lennard jones full atom repulsive forces',
                       'direction': None, 'function': None, 'filter': None},
                  'fa_sol':
                      {'description': 'Rosetta Energy Term - full atom solvent forces',
                       'direction': None, 'function': None, 'filter': None},
                  'hbond_bb_sc':
                      {'description': 'Rosetta Energy Term - backbone/sidechain hydrogen bonding',
                       'direction': None, 'function': None, 'filter': None},
                  'hbond_lr_bb':
                      {'description': 'Rosetta Energy Term - long range backbone hydrogen bonding',
                       'direction': None, 'function': None, 'filter': None},
                  'hbond_sc':
                      {'description': 'Rosetta Energy Term - side-chain hydrogen bonding',
                       'direction': None, 'function': None, 'filter': None},
                  'hbond_sr_bb':
                      {'description': 'Rosetta Energy Term - short range backbone hydrogen bonding',
                       'direction': None, 'function': None, 'filter': None},
                  'lk_ball_wtd':
                      {'description': 'Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
                       'direction': None, 'function': None, 'filter': None},
                  'omega':
                      {'description': 'Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
                       'direction': None, 'function': None, 'filter': None},
                  'p_aa_pp':
                      {'description': '"Rosetta Energy Term - statistical probability of an amino acid given angles phi'
                          , 'direction': None, 'function': None, 'filter': None},
                  'pro_close':
                      {'description': 'Rosetta Energy Term - to favor closing of proline rings',
                       'direction': None, 'function': None, 'filter': None},
                  'rama_prepro':
                      {'description': 'Rosetta Energy Term - amino acid dependent term to favor certain Ramachandran '
                                      'angles on residue before proline',
                       'direction': None, 'function': None, 'filter': None},
                  'yhh_planarity':
                      {'description': 'Rosetta Energy Term - favor planarity of tyrosine alcohol hydrogen',
                       'direction': None, 'function': None, 'filter': None}}

nanohedra_metrics = ['nanohedra_score_normalized', 'nanohedra_score_center_normalized', 'nanohedra_score',
                     'nanohedra_score_center', 'number_fragment_residues_total', 'number_fragment_residues_center',
                     'multiple_fragment_ratio', 'percent_fragment_helix', 'percent_fragment_strand',
                     'percent_fragment_coil', 'number_of_fragments', 'total_interface_residues',
                     'total_non_fragment_interface_residues', 'percent_residues_fragment_total',
                     'percent_residues_fragment_center']
# These metrics are necessary for all calculations performed during the analysis script. If missing, something will fail
necessary_metrics = {'buns_complex', 'buns_1_unbound', 'contact_count', 'coordinate_constraint',
                     'favor_residue_energy', 'hbonds_res_selection_complex', 'hbonds_res_selection_1_bound',
                     'interface_connectivity_1',
                     'interface_separation', 'interface_energy_1_bound',
                     'interface_energy_1_unbound',  'interface_energy_complex',
                     'interaction_energy_complex', groups, 'rosetta_reference_energy', 'shape_complementarity',
                     'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
                     'sasa_hydrophobic_1_bound', 'sasa_polar_1_bound',
                     'sasa_total_1_bound', 'solvation_energy_complex', 'solvation_energy_1_bound',
                     'solvation_energy_1_unbound'
                     }
#                      'buns_2_unbound',
#                      'hbonds_res_selection_2_bound', 'interface_connectivity_2',
#                      'interface_energy_2_bound', 'interface_energy_2_unbound',
#                      'sasa_hydrophobic_2_bound', 'sasa_polar_2_bound', 'sasa_total_2_bound',
#                      'solvation_energy_2_bound', 'solvation_energy_2_unbound',
#                      }
#                    'rmsd'
#                      'buns_asu_hpol', 'buns_nano_hpol', 'buns_asu', 'buns_nano', 'buns_total',
#                      'fsp_total_stability', 'full_stability_complex',
#                      'number_hbonds', 'total_interface_residues',
#                      'average_fragment_z_score', 'nanohedra_score', 'number_of_fragments', 'interface_b_factor_per_res',
# unused, just a placeholder for the metrics in population
final_metrics = {'interface_buried_hbonds', 'contact_count', 'core', 'coordinate_constraint',
                 'divergence_design_per_residue', 'divergence_evolution_per_residue', 'divergence_fragment_per_residue',
                 'divergence_interface_per_residue', 'favor_residue_energy',
                 'interface_area_hydrophobic', 'interface_area_polar', 'interface_area_total',
                 'interface_composition_similarity', 'interface_connectivity_1', 'interface_connectivity_2',
                 'interface_energy_1_unbound', 'interface_energy_2_unbound',
                 'interface_energy_complex', 'interface_energy', 'interface_energy_unbound',
                 'int_separation',
                 'interaction_energy_complex', 'interface_b_factor_per_res', 'multiple_fragment_ratio',
                 'number_hbonds', 'nanohedra_score', 'nanohedra_score_center',
                 'nanohedra_score_per_res', 'nanohedra_score_center_per_res_center',
                 'number_fragment_residues_total', 'number_fragment_residues_central', 'number_of_fragments',
                 'observations',
                 'observed_design', 'observed_evolution', 'observed_fragment', 'observed_interface', 'percent_core',
                 'percent_fragment', 'percent_fragment_coil', 'percent_fragment_helix', 'percent_fragment_strand',
                 'percent_interface_area_hydrophobic', 'percent_interface_area_polar',
                 'percent_residues_fragment_center',
                 'percent_residues_fragment_total', 'percent_rim', 'percent_support',
                 'protocol_energy_distance_sum', 'protocol_similarity_sum', 'protocol_seq_distance_sum',
                 'rosetta_reference_energy', 'rim', 'rmsd', 'shape_complementarity', 'solvation_energy', 'support',
                 'symmetry', 'total_non_fragment_interface_residues'}
#                  'buns_heavy_total', 'buns_hpol_total', 'buns_total',
#                These are missing the bb_hb contribution and are inaccurate
#                   'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex',
#                   'int_energy_context_delta', 'int_energy_context_oligomer',
#                These are accounted for in other pose metrics
#                   'nanohedra_score', 'average_fragment_z_score', 'number_of_fragments', 'total_interface_residues',
#                   'interface_b_factor_per_res'}
#                These could be added in, but seem to be unnecessary
#                   'fsp_total_stability', 'full_stability_complex',

columns_to_rename = {'shape_complementarity_median_dist': 'interface_separation',
                     'ref': 'rosetta_reference_energy',
                     # 'relax_switch': groups, 'no_constraint_switch': groups, 'limit_to_profile_switch': groups,
                     # 'combo_profile_switch': groups, 'design_profile_switch': groups,
                     # 'favor_profile_switch': groups, 'consensus_design_switch': groups,
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
                     # 'R_int_connectivity_1': 'interface_connectivity_1',
                     # 'R_int_connectivity_2': 'interface_connectivity_2',
                     }
#                      'total_score': 'REU', 'decoy': 'design', 'symmetry_switch': 'symmetry',

clean_up_intermediate_columns = ['int_energy_no_intra_residue_score',  # 'interface_energy_bound',
                                 'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
                                 'sasa_hydrophobic_bound', 'sasa_hydrophobic_1_bound', 'sasa_hydrophobic_2_bound',
                                 'sasa_polar_bound', 'sasa_polar_1_bound', 'sasa_polar_2_bound',
                                 'sasa_total_bound', 'sasa_total_1_bound', 'sasa_total_2_bound',
                                 # 'buns_complex', 'buns_unbound', 'buns_1_unbound', 'buns_2_unbound',
                                 'solvation_energy_1_bound', 'solvation_energy_2_bound', 'solvation_energy_1_unbound',
                                 'solvation_energy_2_unbound',
                                 'interface_energy_1_bound', 'interface_energy_1_unbound', 'interface_energy_2_bound',
                                 'interface_energy_2_unbound',
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
               'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex',
               # 'buns_asu', 'buns_asu_hpol', 'buns_nano', 'buns_nano_hpol', 'buns_total',
               'angle_constraint', 'atom_pair_constraint', 'chainbreak', 'coordinate_constraint', 'dihedral_constraint',
               'metalbinding_constraint', 'rmsd', 'repack_switch', 'sym_status'
               ]
# remove_score_columns = ['hbonds_res_selection_asu', 'hbonds_res_selection_unbound']
#                'full_stability_oligomer_A', 'full_stability_oligomer_B']

# columns_to_remove = ['decoy', 'symmetry_switch', 'metrics_symmetry', 'oligomer_switch', 'total_score',
#                      'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex']

# subtract columns using tuple [0] - [1] to make delta column
delta_pairs = {'interface_buried_hbonds': ('buns_complex', 'buns_unbound'),
               'interface_energy': ('interface_energy_complex', 'interface_energy_unbound'),
               # 'interface_energy_no_intra_residue_score': ('interface_energy_complex', 'interface_energy_bound'),
               'interface_bound_activation_energy': ('interface_energy_bound', 'interface_energy_unbound'),
               'solvation_energy': ('solvation_energy_unbound', 'solvation_energy_complex'),
               # 'solvation_energy': ('interaction_energy_complex', 'interface_energy_no_intra_residue_score'),
               'interface_area_hydrophobic': ('sasa_hydrophobic_bound', 'sasa_hydrophobic_complex'),
               'interface_area_polar': ('sasa_polar_bound', 'sasa_polar_complex'),
               'interface_area_total': ('sasa_total_bound', 'sasa_total_complex')
               }
#                'int_energy_context_delta': ('int_energy_context_complex', 'int_energy_context_oligomer'),
#                'full_stability_delta': ('full_stability_complex', 'full_stability_oligomer')}
#                'number_hbonds': ('hbonds_res_selection_complex', 'hbonds_oligomer')}


# divide columns using tuple [0] / [1] to make divide column
division_pairs = {'percent_interface_area_hydrophobic': ('interface_area_hydrophobic', 'interface_area_total'),
                  'percent_interface_area_polar': ('interface_area_polar', 'interface_area_total'),
                  'percent_core': ('core', 'total_interface_residues'),
                  'percent_rim': ('rim', 'total_interface_residues'),
                  'percent_support': ('support', 'total_interface_residues'),
                  'buns_per_ang': ('buns_total', 'interface_area_total'),
                  'interface_energy_density': ('interface_energy', 'interface_area_total')}

# All Rosetta based score terms ref is most useful to keep for whole pose to give "unfolded ground state"
rosetta_terms = ['lk_ball_wtd', 'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'yhh_planarity', 'dslf_fa13',
                 'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'ref']

# Current protocols in use in design.xml
protocols = ['design_profile_switch', 'favor_profile_switch', 'limit_to_profile_switch', 'no_constraint_switch']
protocols_of_interest = {'design_profile', 'no_constraint'}
# protocols_of_interest = ['combo_profile', 'limit_to_profile', 'no_constraint']  # Used for P432 models

# Specific columns of interest to distinguish between design trajectories
significance_columns = ['interface_buried_hbonds',
                        'contact_count', 'interface_energy', 'interface_area_total', 'number_hbonds',
                        'percent_interface_area_hydrophobic', 'shape_complementarity', 'solvation_energy'
#                         'buns_total',
                        ]
# sequence_columns = ['divergence_evolution_per_residue', 'divergence_fragment_per_residue',
#                     'observed_evolution', 'observed_fragment']

stats_metrics = ['mean', 'std']
residue_classificiation = ['core', 'rim', 'support']  # 'hot_spot'
# per_res_keys = ['jsd', 'des_jsd', 'int_jsd', 'frag_jsd']

# from table 1, theoretical values of Tien et al. 2013
residue_template = {'energy': {'complex': 0., 'unbound': 0., 'fsp': 0., 'cst': 0.},
                    'sasa': {'total': {'complex': 0., 'unbound': 0.}, 'polar': {'complex': 0., 'unbound': 0.},
                             'hydrophobic': {'complex': 0., 'unbound': 0.}},
                    'type': None, 'core': 0, 'rim': 0, 'support': 0, 'interior': 0, 'hbond': 0}  # , 'hot_spot': 0}
fragment_metric_template = {'center_residues': set(), 'total_residues': set(),
                            'nanohedra_score': 0.0, 'nanohedra_score_center': 0.0, 'multiple_fragment_ratio': 0.0,
                            'number_fragment_residues_total': 0, 'number_fragment_residues_center': 0,
                            'number_fragments': 0, 'percent_fragment_helix': 0.0, 'percent_fragment_strand': 0.0,
                            'percent_fragment_coil': 0.0}


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
                # to ensure old trajectories don't have lingering protocol info
                for protocol in protocols:
                    if protocol in entry:  # ensure that the new scores has a protocol before removing the old one.
                        for rm_protocol in protocols:
                            score_dict[design].pop(rm_protocol, None)
                score_dict[design].update(entry)

    return score_dict


def keys_from_trajectory_number(pdb_dict):
    """Remove all string from dictionary keys except for string after last '_'. Ex 'design_0001' -> '0001'

    Returns:
        (dict): {cleaned_key: value, ...}
    """
    return {key.split('_')[-1]: value for key, value in pdb_dict.items()}


def join_columns(x):
    """Combine columns in a dataframe with the same column name. Keep only the last column record"""
    new_data = ','.join(x[x.notnull()].astype(str))
    return new_data.split(',')[-1]


def columns_to_new_column(df, column_dict, mode='add'):
    """Set new column value by taking an operation of one column on another

    Can perform summation and subtraction if a set of columns is provided
    Args:
        df (pandas.DataFrame): Dataframe where the columns are located
        column_dict (dict[mapping[str,tuple]]): Keys are new column names, values are tuple of existing columns where
        value[0] mode(operation) value[1] = df[key]
    Keyword Args:
        mode='add' (str) = What operator to use?
            Viable options are included in module operator, but could be 'sub', 'mul', 'truediv', etc.
    Returns:
        (pandas.DataFrame): Dataframe with new column values
    """
    for new_column, column_set in column_dict.items():
        try:
            df[new_column] = operator.attrgetter(mode)(operator)(df[column_set[0]], df[column_set[1]])
        except KeyError:
            pass
        if len(column_set) > 2 and mode in ['add', 'sub']:  # >2 values in set, perform repeated operations Ex: SUM, SUB
            for iteration in enumerate(column_set[2:], 2):  # perform an iteration for every N-2 items in the column_set
                try:
                    df[new_column] = operator.attrgetter(mode)(operator)(df[new_column], df[column_set[iteration]])
                except KeyError:
                    pass

    return df


def calc_relative_sa(aa, sa):
    """Calculate relative surface area according to theoretical values of Tien et al. 2013"""
    return round(sa / gxg_sasa[aa], 2)


def hbond_processing(score_dict, columns, offset=None):
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering
    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        columns (list): ['hbonds_res_selection_complex', 'hbonds_res_selection_1_unbound',
            'hbonds_res_selection_2_unbound']
    Keyword Args:
        offset=None (dict[mapping[int, int]]): {1: 0, 2: 102, ...} The amount to offset each chain by
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
                if column.split('_')[-1] == 'unbound' and offset:  # 'oligomer'
                    res_offset = offset[column.split('_')[-2]]
                for i in range(len(hbonds)):
                    hbonds[i] = int(hbonds[i][:-1]) + res_offset  # remove chain ID off last index
            entry_dict[column.split('_')[3]] = set(hbonds)
        if len(entry_dict) == 3:
            hbond_dict[entry] = list((entry_dict['complex'] - entry_dict['1']) - entry_dict['2'])
            #                                                   entry_dict['A']) - entry_dict['B'])
        else:
            hbond_dict[entry] = list()
        #     logger.error('%s: Missing hbonds_res_selection_ data for %s. Hbonds inaccurate!' % (pose, entry))

    return hbond_dict


def dirty_hbond_processing(score_data, offset=None):  # columns
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering
    Args:
        score_data (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...',
                            'hbonds_res_selection_1_bound': '26A'}, ...}
    Keyword Args:
        offset=None (dict[mapping[int, int]]): {1: 0, 2: 102, ...} The amount to offset each chain by
    Returns:
        (dict[mapping[str, set]]): {'0001': [34, 54, 67, 68, 106, 178], ...}
    """
    hbond_dict = {}
    res_offset = 0
    for entry, data in score_data.items():
        unbound_bonds, complex_bonds = set(), set()
        # hbonds_entry = []
        # for column in columns:
        for column, value in data.items():
            if column.startswith('hbonds_res_selection'):
                hbonds = value.split(',')
                meta_data = column.split('_')  # ['hbonds', 'res', 'selection', 'complex/interface_number', '[unbound]']
                # ensure there are hbonds present
                # if hbonds[0] == '' and len(hbonds) == 1:
                parsed_hbonds = set()
                # else:
                if meta_data[-1] == 'bound' and offset:  # find offset according to chain
                    res_offset = offset[meta_data[-2]]
                for i in range(len(hbonds)):
                    # hbonds[i] = res_offset + int(hbonds[i][:-1])  # remove chain ID off last index of string
                    parsed_hbonds.add(res_offset + int(hbonds[i][:-1]))  # remove chain ID off last index of string
                if meta_data[3] == 'complex':
                    complex_bonds = parsed_hbonds
                else:  # from another state
                    unbound_bonds.union(parsed_hbonds)
        if complex_bonds:  # 'complex', '1', '2'
            hbond_dict[entry] = complex_bonds.difference(unbound_bonds)
            # hbond_dict[entry] = [hbonds_entry['complex'].difference(hbonds_entry['1']).difference(hbonds_entry['2']))]
            #                                                         hbonds_entry['A']).difference(hbonds_entry['B'])
        else:  # no hbonds were found in the complex
            hbond_dict[entry] = set()
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


def interface_residue_composition_similarity(series):
    """Calculate the composition difference for pose residue classification

    Args:
        series (union[pandas.Series, dict]): Container with 'interface_area_total', 'core', 'rim', and 'support' keys
    Returns:
        (float): Average similarity for expected residue classification given the observed classification
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

    class_ratio_diff_d = {}
    for residue_class, function in classification_fxn_d.items():
        expected = function(int_area)
        class_ratio_diff_d[residue_class] = (1 - (abs(series[residue_class] - expected) / expected))
        if class_ratio_diff_d[residue_class] < 0:
            # above calculation fails to bound between 0 and 1 with large obs values due to proportion > 1
            class_ratio_diff_d[residue_class] = 0

    return sum(class_ratio_diff_d.values()) / len(class_ratio_diff_d)


def residue_processing(score_dict, mutations, columns, offset=None, hbonds=None):
    """Process Residue Metrics from Rosetta score dictionary

    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        mutations (dict): {'0001': {mutation_index: {'from': 'A', 'to: 'K'}, ...}, ...}
        columns (list): ['per_res_energy_complex_5', 'per_res_sasa_polar_1_unbound_5', 
            'per_res_energy_1_unbound_5', ...]
    Keyword Args:
        offset=None (dict[mapping[int, int]]): {1: 0, 2: 102, ...} The amount to offset each chain by
        hbonds=None (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    Returns:
        residue_dict (dict): {'0001': {15: {'type': 'T', 'energy_delta': -2.771, 'bsa_polar': 13.987, 'bsa_hydrophobic': 
            22.29, 'bsa_total': 36.278, 'hbond': 0, 'core': 0, 'rim': 1, 'support': 0}, ...}, ...}
    """  # , 'hot_spot': 1
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
            pose_state = metadata[-2]  # unbound or complex or fsp (favor_sequence_profile)  # cst (constraint)
            if pose_state == 'unbound' and offset:  # 'oligomer'
                res += offset[metadata[-3]]  # get interface, chain name, length offset
            if res not in residue_dict:
                residue_dict[res] = deepcopy(residue_template)
            if r_type == 'sasa':
                # Ex. per_res_sasa_hydrophobic_1_unbound_15 or per_res_sasa_hydrophobic_complex_15
                polarity = metadata[3]  # hydrophobic or polar or total
                residue_dict[res][r_type][polarity][pose_state] = round(score_dict[entry][column], 3)
            else:
                # Ex. per_res_energy_1_unbound_15 or per_res_energy_complex_15
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
                - residue_dict[res]['energy']['unbound']  # - residue_dict[res]['energy']['cst']  # 'oligomer'
            # - residue_dict[res]['energy']['fsp']
            rel_oligomer_sasa = calc_relative_sa(residue_dict[res]['type'],
                                                 residue_dict[res]['sasa']['total']['unbound'])  # 'oligomer'
            rel_complex_sasa = calc_relative_sa(residue_dict[res]['type'],
                                                residue_dict[res]['sasa']['total']['complex'])
            for polarity in residue_dict[res]['sasa']:
                # convert sasa measurements into bsa measurements
                residue_dict[res]['bsa_' + polarity] = round(
                    residue_dict[res]['sasa'][polarity]['unbound'] - residue_dict[res]['sasa'][polarity][
                        'complex'], 2)  # 'oligomer'
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


def dirty_residue_processing(score_dict, mutations, offset=None, hbonds=None):  # pose_length,
    """Process Residue Metrics from Rosetta score dictionary

    One-indexed residues
    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15A': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...'}, ...}
        mutations (dict): {'reference': {mutation_index: {'from': 'A', 'to: 'K'}, ...},
                           '0001': {mutation_index: {}, ...}, ...}
    Keyword Args:
        offset=None (dict[mapping[int, int]]): {1: 0, 2: 102, ...} The amount to offset each chain by
        hbonds=None (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    Returns:
        (dict): {'0001': {15: {'type': 'T', 'energy_delta': -2.771, 'bsa_polar': 13.987, 'bsa_hydrophobic': 22.29,
                               'bsa_total': 36.278, 'hbond': 0, 'core': 0, 'rim': 1, 'support': 0},  # , 'hot_spot': 1
                          ...}, ...}
    """
    # pose_length (int): The number of residues in the pose
    pose_length = len(mutations['reference'])
    warn = False
    total_residue_dict = {}
    for design, scores in score_dict.items():
        residue_data = {}
        # for column in columns:
        for key, value in scores.items():
            # metadata = column.split('_')
            if key.startswith('per_res_'):
                metadata = key.split('_')
                # residue_number = int(metadata[-1])
                # residue_number = int(metadata[-1][:-1])  # remove the chain identifier used with rosetta_numbering="False"
                residue_number = int(metadata[-1].translate(digit_translate_table))  # remove chain_id in rosetta_numbering="False"
                if residue_number > pose_length:
                    if not warn:  # TODO this can move to residue_processing (clean) once instated
                        warn = True
                        logger.warning('Encountered %s which has residue number > the pose length (%d). Scores above '
                                       'will be discarded. Use pbd_numbering on all Rosetta PerResidue SimpleMetrics to'
                                       ' ensure that symmetric copies have the same residue number on symmetry mates.'
                                       % (key, pose_length))
                    continue
                metric = metadata[2]  # energy or sasa
                pose_state = metadata[-2]  # unbound or complex
                if pose_state == 'unbound' and offset:
                    residue_number += offset[metadata[-3]]  # get oligomer chain offset
                if residue_number not in residue_data:
                    residue_data[residue_number] = deepcopy(residue_template)
                if metric == 'sasa':
                    # Ex. per_res_sasa_hydrophobic_1_unbound_15 or per_res_sasa_hydrophobic_complex_15
                    polarity = metadata[3]
                    residue_data[residue_number][metric][polarity][pose_state] = value
                else:
                    # Ex. per_res_energy_1_unbound_15 or per_res_energy_complex_15
                    residue_data[residue_number][metric][pose_state] += value
        # if residue_data:
        for residue_number, data in residue_data.items():
            try:
                data['type'] = mutations[design][residue_number]  # % pose_length]
            except KeyError:
                data['type'] = mutations['reference'][residue_number]  # % pose_length]  # fill with aa from wt seq
            if hbonds:
                if residue_number in hbonds[design]:
                    data['hbond'] = 1
            data['energy_delta'] = round(data['energy']['complex'] - data['energy']['unbound'], 2)
            #     - data['energy']['fsp'] - data['energy']['cst']
            # because Rosetta energy is from unfavored/unconstrained scorefunction, we don't need to subtract
            relative_oligomer_sasa = calc_relative_sa(data['type'], data['sasa']['total']['unbound'])
            relative_complex_sasa = calc_relative_sa(data['type'], data['sasa']['total']['complex'])
            for polarity in data['sasa']:
                # convert sasa measurements into bsa measurements
                data['bsa_%s' % polarity] = \
                    round(data['sasa'][polarity]['unbound'] - data['sasa'][polarity]['complex'], 2)

            if data['bsa_total'] > 0:
                if relative_oligomer_sasa < 0.25:
                    data['support'] = 1
                elif relative_complex_sasa < 0.25:
                    data['core'] = 1
                else:
                    data['rim'] = 1
            else:  # Todo remove residue from dict as no design should be done? keep interior residue constant?
                if relative_complex_sasa < 0.25:
                    data['interior'] = 1
                # else:
                #     residue_data[residue_number]['surface'] = 1
            data['coordinate_constraint'] = round(data['energy']['cst'], 2)
            data['residue_favored'] = round(data['energy']['fsp'], 2)
            data.pop('energy')
            data.pop('sasa')
            # if residue_data[residue_number]['energy'] <= hot_spot_energy:
            #     residue_data[residue_number]['hot_spot'] = 1
        total_residue_dict[design] = residue_data

    return total_residue_dict


def mutation_conserved(residue_info, bkgnd):
    """Process residue mutations compared to evolutionary background. Returns 1 if residue is observed in background

    Both residue_dict and background must be same index
    Args:
        residue_info (dict): {15: {'type': 'T', ...}, ...}
        bkgnd (dict): {0: {'A': 0, 'R': 0, ...}, ...}
    Returns:
        conservation_dict (dict): {15: 1, 21: 0, 25: 1, ...}
    """
    return {res: 1 if bkgnd[res][info['type']] > 0 else 0 for res, info in residue_info.items() if res in bkgnd}
    # conservation_dict = {}
    # for residue, info in residue_info.items():
    #     residue_background = background.get(residue, None)
    #     if not residue_background:
    #         continue
    #     if residue_background[info['type']] > 0:
    #         conservation_dict[residue] = 1
    #     else:
    #         conservation_dict[residue] = 0
    #
    # return conservation_dict


def per_res_metric(sequence_metrics, key=None):
    """Find metric value/residue in sequence dictionary specified by key

    Args:
        sequence_metrics (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    Keyword Args:
        key='jsd' (str): Name of the residue metric to average
    Returns:
        jsd_per_res (float): 0.367
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


def hydrophobic_collapse_index(sequence, hydrophobicity='standard'):  # TODO Validate
    """Calculate hydrophobic collapse index for a particular sequence of an iterable object and return a HCI array

    """
    seq_length = len(sequence)
    lower_range, upper_range, range_correction = 3, 9, 1
    range_size = upper_range - lower_range  # + range_correction
    yes = 1
    no = 0
    if hydrophobicity == 'expanded':
        hydrophobic = ['F', 'I', 'L', 'M', 'V', 'W', 'Y']
    else:  # hydrophobicity == 'standard':
        hydrophobic = ['F', 'I', 'L', 'V']

    sequence_array = [yes if aa in hydrophobic else no for aa in sequence]
    # for aa in sequence:
    #     if aa in hydrophobic:
    #         sequence_array.append(yes)
    #     else:
    #         sequence_array.append(no)

    # make an array with # of rows equal to upper range (+1 for indexing), length equal to # of letters in sequence
    window_array = np.zeros((range_size, seq_length))  # [[0] * (seq_length + 1) for i in range(upper_range + 1)]
    for idx, j in enumerate(range(lower_range, upper_range + range_correction), 0):
        # iterate over the window range
        window_len = math.floor(j / 2)
        modulus = j % 2
        # check if the range is odd or even, then calculate score accordingly, with cases for N- and C-terminal windows
        if modulus == 1:  # range is odd
            for k in range(seq_length):
                s = 0
                if k < window_len:  # N-terminus
                    for n in range(k + window_len + range_correction):
                        s += sequence_array[n]
                elif k + window_len >= seq_length:  # C-terminus
                    for n in range(k - window_len, seq_length):
                        s += sequence_array[n]
                else:
                    for n in range(k - window_len, k + window_len + range_correction):
                        s += sequence_array[n]
                window_array[idx][k] = s / j
        else:  # range is even
            for k in range(seq_length):
                s = 0
                if k < window_len:  # N-terminus
                    for n in range(k + window_len + range_correction):
                        if n == k + window_len:
                            s += 0.5 * sequence_array[n]
                        else:
                            s += sequence_array[n]
                elif k + window_len >= seq_length:  # C-terminus
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
                window_array[idx][k] = s / j
    logger.debug(window_array)
    # return the mean score for each sequence position
    return window_array.mean(axis=0)
    # hci = np.zeros(seq_length)  # [0] * (seq_length + 1)
    # for k in range(seq_length):
    #     for j in range(lower_range, upper_range + range_correction):
    #         hci[k] += window_array[j][k]
    #     hci[k] /= range_size
    #     hci[k] = round(hci[k], 3)

    # return hci


def calculate_column_number(num_groups=1, misc=0, sig=0):  # UNUSED, DEPRECIATED
    total = len(final_metrics) * len(stats_metrics)
    total += len(significance_columns) * num_groups * len(stats_metrics)
    total += misc
    total += sig  # for protocol pair mean similarity measure
    total += sig * len(significance_columns)  # for protocol pair individual similarity measure

    return total


def df_filter_index_by_value(df, **kwargs):
    """Take a df and retrieve the indices which have column values greater_equal/less_equal to a value depending
    on whether the column should be sorted max/min

    Args:
        df (pandas.DataFrame): DataFrame to filter indices on
    Keyword Args:
        kwargs (dict): {column: {'direction': 'min', 'value': 0.3}, ...}
    Returns:
        (dict): {column: {'direction': 'min', 'value': 0.3, 'idx': ['0001', '0002', ...]}, ...}
    """
    for idx in kwargs:
        if kwargs[idx]['direction'] == 'max':
            kwargs[idx]['idx'] = df[df[idx] >= kwargs[idx]['value']].index.to_list()
        if kwargs[idx]['direction'] == 'min':
            kwargs[idx]['idx'] = df[df[idx] <= kwargs[idx]['value']].index.to_list()

    return kwargs


def filter_pose(df_file, filter=None, weight=None, consensus=False, sort_df=True):
    """Return the indices from a dataframe that pass an set of filters (optional) and are ranked according to weight as
    specified by user values.

    Args:
        df_file (union[str, pandas.DataFrame]): DataFrame to filter/weight indices
    Keyword Args:
        filter=False (bool): Whether filters are going to remove viable candidates
        weight=False (bool): Whether weights are going to select the poses
        consensus=False (bool): Whether consensus designs should be chosen
    Returns:
        (pandas.DataFrame): The dataframe of selected designs based on the provided filters and weights
    """
    idx_slice = pd.IndexSlice
    # Grab pose info from the DateFrame and drop all classifiers in top two rows.
    if isinstance(df_file, pd.DataFrame):
        df = df_file
    else:
        df = pd.read_csv(df_file, index_col=0, header=[0, 1, 2])
    logger.info('Number of starting designs = %d' % len(df))
    _df = df.loc[:, idx_slice['pose',
                              df.columns.get_level_values(1) != 'std', :]].droplevel(1, axis=1).droplevel(0, axis=1)

    filter_df = pd.DataFrame(master_metrics)
    if filter:
        available_filters = _df.columns.to_list()
        if isinstance(filter, dict):
            filters = filter
        else:
            filters = query_user_for_metrics(available_filters, mode='filter', level='design')
        logger.info('Using filter parameters: %s' % str(filters))

        # When df is not ranked by percentage
        _filters = {metric: {'direction': filter_df.loc['direction', metric], 'value': value}
                    for metric, value in filters.items()}

        # Filter the DataFrame to include only those values which are le/ge the specified filter
        filters_with_idx = df_filter_index_by_value(_df, **_filters)
        filtered_indices = {metric: filters_with_idx[metric]['idx'] for metric in filters_with_idx}
        logger.info('Number of designs passing filters:\n\t%s'
                    % '\n\t'.join('%6d - %s' % (len(indices), metric) for metric, indices in filtered_indices.items()))
        final_indices = index_intersection(filtered_indices)
        logger.info('Final set of designs passing all filters has %d members' % len(final_indices))
        if len(final_indices) == 0:
            raise DesignError('There are no poses left after filtering! Try choosing less stringent values or make '
                              'better designs!')
        _df = _df.loc[final_indices, :]

    # When df IS ranked by percentage
    # bottom_percent = (num_designs / len(df))
    # top_percent = 1 - bottom_percent
    # min_max_to_top_bottom = {'min': bottom_percent, 'max': top_percent}
    # _filters = {metric: {'direction': filter_df.loc['direction', metric],
    #                      'value': min_max_to_top_bottom[filter_df.loc['direction', metric]]} for metric in filters}

    # _sort = {metric: {'direction': filter_df.loc['direction', metric],
    #                   'value': min_max_to_top_bottom[filter_df.loc['direction', metric]]} for metric in sort_s.index}
    # filters_with_idx = df_filter_index_by_value(ranked_df, **_sort)

    if consensus:
        protocol_df = df.loc[:, idx_slice['consensus', ['mean', 'stats'], :]].droplevel(1, axis=1)
        #     df.loc[:, idx_slice[df.columns.get_level_values(0) != 'pose', ['mean', 'stats'], :]].droplevel(1, axis=1)
        # stats_protocol_df = \
        #     df.loc[:, idx_slice[df.columns.get_level_values(0) != 'pose', df.columns.get_level_values(1) == 'stats',
        #     :]].droplevel(1, axis=1)
        # design_protocols_df = pd.merge(protocol_df, stats_protocol_df, left_index=True, right_index=True)
        # TODO make more robust sampling from specific protocol
        _df = pd.merge(protocol_df.loc[:, idx_slice['consensus', :]],
                       df.droplevel(0, axis=1).loc[:, idx_slice[:, 'percent_fragment']],
                       left_index=True, right_index=True).droplevel(0, axis=1)
    # filtered_indices = {}

    # for metric in filters:
    #     filtered_indices[metric] = set(df[df.droplevel(0, axis=1)[metric] >= filters[metric]].index.to_list())
    #     logger.info('Number of designs passing %s = %d' % (metric, len(filtered_indices[metric])))
    # ranked_df = _df.rank(method='min', pct=True, )  # default is to rank lower values as closer to 1
    # need {column: {'direction': 'max', 'value': 0.5, 'idx_slice': []}, ...}

    # only used to check out the number of designs in each filter
    # for _filter in crystal_filters_with_idx:
    #     print('%s designs = %d' % (_filter, len(crystal_filters_with_idx[_filter]['idx_slice'])))

    # {column: {'direction': 'min', 'value': 0.3, 'idx_slice': ['0001', '0002', ...]}, ...}
    if weight:
        # display(ranked_df[weights_s.index.to_list()] * weights_s)
        available_metrics = _df.columns.to_list()
        if isinstance(weight, dict):
            weights = weight
        else:
            weights = query_user_for_metrics(available_metrics, mode='weight', level='design')
        logger.info('Using weighting parameters: %s' % str(weights))
        _weights = {metric: {'direction': filter_df.loc['direction', metric], 'value': value}
                    for metric, value in weights.items()}
        weight_direction = {'max': True, 'min': False}  # max - ascending=False, min - ascending=True
        # weights_s = pd.Series(weights)
        weight_score_s_d = {}
        for metric in _weights:
            weight_score_s_d[metric] = \
                _df[metric].rank(ascending=weight_direction[_weights[metric]['direction']],
                                 method=_weights[metric]['direction'], pct=True) * _weights[metric]['value']

        design_score_df = pd.concat([weight_score_s_d[weight] for weight in weights], axis=1)
        if sort_df:
            weighted_s = design_score_df.sum(axis=1).sort_values(ascending=False)
        else:
            weighted_s = design_score_df.sum(axis=1)
        weighted_df = pd.concat([weighted_s], keys=[('pose', 'sum', 'selection_weight')], axis=1)
        final_df = pd.merge(weighted_df, df, left_index=True, right_index=True)
        # designs = weighted_s.index.to_list()
    else:
        final_df = df.loc[_df.index.to_list(), :]
        # designs = _df.index.to_list()
    # these will be sorted by the largest value to the smallest
    # design_scores_s = (ranked_df[weights_s.index.to_list()] * weights_s).sum(axis=1).sort_values(ascending=False)
    # designs = design_scores_s.index.to_list()
    # designs = design_scores_s.index.to_list()[:num_designs]
    # return designs
    return final_df


def calculate_match_metrics(fragment_matches):
    """Return the various metrics calculated by overlapping fragments at the interface of two proteins

    Args:
        fragment_matches (list[dict]): [{'mapped': entity1_resnum, 'match': score_term, 'paired': entity2_resnum,
                                         'culster': cluster_id}, ...]
    Returns:
        (dict): {'mapped': {'center': {'residues': (set[int]), 'score': (float), 'number': (int)},
                            'total': {'residues': (set[int]), 'score': (float), 'number': (int)},
                            'match_scores': {residue number(int): (list[score (float)]), ...},
                            'index_count': {index (int): count (int), ...},
                            'multiple_ratio': (float)}
                 'paired': {'center': , 'total': , 'match_scores': , 'index_count': , 'multiple_ratio': },
                 'total':  {'center': {'score': , 'number': },
                            'total': {'score': , 'number': },
                            'index_count': , 'multiple_ratio': , 'observations': (int)}
                 }
        # (tuple): all_residue_score (Nanohedra), center_residue_score, total_residues_with_fragment_overlap, \
        # central_residues_with_fragment_overlap, multiple_frag_ratio, total_fragment_content
    """
    if not fragment_matches:
        # raise DesignError('No fragment matches were passed! Can\'t calculate match metrics')
        return None

    fragment_i_index_count_d = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    fragment_j_index_count_d = copy(fragment_i_index_count_d)
    total_fragment_content = copy(fragment_i_index_count_d)

    entity1_center_match_scores, entity2_center_match_scores = {}, {}
    entity1_match_scores, entity2_match_scores = {}, {}
    separated_fragment_metrics = {'mapped': {'center': {'residues': set()}, 'total': {'residues': set()}},
                                  'paired': {'center': {'residues': set()}, 'total': {'residues': set()}},
                                  'total': {'observations': len(fragment_matches), 'center': {}, 'total': {}}}
    for fragment in fragment_matches:
        center_resnum1, center_resnum2, match_score = fragment['mapped'], fragment['paired'], fragment['match']
        separated_fragment_metrics['mapped']['center']['residues'].add(center_resnum1)
        separated_fragment_metrics['paired']['center']['residues'].add(center_resnum2)

        # TODO an ideal measure of the central importance would weight central fragment observations to new center
        #  fragments higher than repeated observations. Say mapped residue 45 to paired 201. If there are two
        #  observations of this pair, just different ijk indices, then these would take the SUMi->n 1/i*2 additive form.
        #  If the observation went from residue 45 to paired residue 204, they constitute separate, but less important
        #  observations as these are part of the same SS and it is implied that if 201 contacts, 204 (and 198) are
        #  possible. In the case where the interaction goes from a residue to a different secondary structure, then
        #  this is the most important, say residue 45 to residue 322 on a separate helix. This indicates that
        #  residue 45 is ideally placed to interact with a separate SS and add them separately to the score
        #  |
        #  How the data structure to build this relationship looks is likely a graph, which has significant overlap to
        #  work I did on the consensus sequence higher order relationships in 8/20. I may find some ground in the middle
        #  of these two ideas where the graph theory could take hold and a more useful scoring metric could emerge,
        #  also possibly with a differentiable equation I could relate pose transformations to favorable fragments found
        #  in the interface.
        if center_resnum1 not in entity1_center_match_scores:
            entity1_center_match_scores[center_resnum1] = [match_score]
        else:
            entity1_center_match_scores[center_resnum1].append(match_score)

        if center_resnum2 not in entity2_center_match_scores:
            entity2_center_match_scores[center_resnum2] = [match_score]
        else:
            entity2_center_match_scores[center_resnum2].append(match_score)

        for resnum1, resnum2 in [(fragment['mapped'] + j, fragment['paired'] + j) for j in range(-2, 3)]:
            separated_fragment_metrics['mapped']['total']['residues'].add(resnum1)
            separated_fragment_metrics['paired']['total']['residues'].add(resnum2)

            if resnum1 not in entity1_match_scores:
                entity1_match_scores[resnum1] = [match_score]
            else:
                entity1_match_scores[resnum1].append(match_score)

            if resnum2 not in entity2_match_scores:
                entity2_match_scores[resnum2] = [match_score]
            else:
                entity2_match_scores[resnum2].append(match_score)

        i, j, k = list(map(int, fragment['cluster'].split('_')))
        fragment_i_index_count_d[i] += 1
        fragment_j_index_count_d[j] += 1

    separated_fragment_metrics['mapped']['center_match_scores'] = entity1_center_match_scores
    separated_fragment_metrics['paired']['center_match_scores'] = entity2_center_match_scores
    separated_fragment_metrics['mapped']['match_scores'] = entity1_match_scores
    separated_fragment_metrics['paired']['match_scores'] = entity2_match_scores
    separated_fragment_metrics['mapped']['index_count'] = fragment_i_index_count_d
    separated_fragment_metrics['paired']['index_count'] = fragment_j_index_count_d
    # -------------------------------------------
    # score the interface individually
    mapped_total_score, mapped_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['mapped'])
    paired_total_score, paired_center_score = nanohedra_fragment_match_score(separated_fragment_metrics['paired'])
    # combine
    all_residue_score = mapped_total_score + paired_total_score
    center_residue_score = mapped_center_score + paired_center_score
    # -------------------------------------------
    # Get individual number of CENTRAL residues with overlapping fragments given z_value criteria
    mapped_central_residues_with_fragment_overlap = len(separated_fragment_metrics['mapped']['center']['residues'])
    paired_central_residues_with_fragment_overlap = len(separated_fragment_metrics['paired']['center']['residues'])
    # combine
    central_residues_with_fragment_overlap = \
        mapped_central_residues_with_fragment_overlap + paired_central_residues_with_fragment_overlap
    # -------------------------------------------
    # Get the individual number of TOTAL residues with overlapping fragments given z_value criteria
    mapped_total_residues_with_fragment_overlap = len(separated_fragment_metrics['mapped']['total']['residues'])
    paired_total_residues_with_fragment_overlap = len(separated_fragment_metrics['paired']['total']['residues'])
    # combine
    total_residues_with_fragment_overlap = \
        mapped_total_residues_with_fragment_overlap + paired_total_residues_with_fragment_overlap
    # -------------------------------------------
    # get the individual multiple fragment observation ratio observed for each side of the fragment query
    mapped_multiple_frag_ratio = \
        separated_fragment_metrics['total']['observations'] / mapped_central_residues_with_fragment_overlap
    paired_multiple_frag_ratio = \
        separated_fragment_metrics['total']['observations'] / paired_central_residues_with_fragment_overlap
    # combine
    multiple_frag_ratio = \
        separated_fragment_metrics['total']['observations'] * 2 / central_residues_with_fragment_overlap
    # -------------------------------------------
    # turn individual index counts into paired counts # and percentages <- not accurate if summing later, need counts
    for index, count in separated_fragment_metrics['mapped']['index_count'].items():
        total_fragment_content[index] += count
        # separated_fragment_metrics['mapped']['index'][index_count] = count / separated_fragment_metrics['number']
    for index, count in separated_fragment_metrics['paired']['index_count'].items():
        total_fragment_content[index] += count
        # separated_fragment_metrics['paired']['index'][index_count] = count / separated_fragment_metrics['number']
    # combined
    # for index, count in total_fragment_content.items():
    #     total_fragment_content[index] = count / (separated_fragment_metrics['total']['observations'] * 2)
    # -------------------------------------------
    # if paired:
    separated_fragment_metrics['mapped']['center']['score'] = mapped_center_score
    separated_fragment_metrics['paired']['center']['score'] = paired_center_score
    separated_fragment_metrics['mapped']['center']['number'] = mapped_central_residues_with_fragment_overlap
    separated_fragment_metrics['paired']['center']['number'] = paired_central_residues_with_fragment_overlap
    separated_fragment_metrics['mapped']['total']['score'] = mapped_total_score
    separated_fragment_metrics['paired']['total']['score'] = paired_total_score
    separated_fragment_metrics['mapped']['total']['number'] = mapped_total_residues_with_fragment_overlap
    separated_fragment_metrics['paired']['total']['number'] = paired_total_residues_with_fragment_overlap
    separated_fragment_metrics['mapped']['multiple_ratio'] = mapped_multiple_frag_ratio
    separated_fragment_metrics['paired']['multiple_ratio'] = paired_multiple_frag_ratio
    #     return separated_fragment_metrics
    # else:
    separated_fragment_metrics['total']['center']['score'] = center_residue_score
    separated_fragment_metrics['total']['center']['number'] = central_residues_with_fragment_overlap
    separated_fragment_metrics['total']['total']['score'] = all_residue_score
    separated_fragment_metrics['total']['total']['number'] = total_residues_with_fragment_overlap
    separated_fragment_metrics['total']['multiple_ratio'] = multiple_frag_ratio
    separated_fragment_metrics['total']['index_count'] = total_fragment_content

    return separated_fragment_metrics


def nanohedra_fragment_match_score(fragment_metric_d):
    """Calculate the Nanohedra score from a dictionary with the 'center' residues and 'match_scores'

    Args:
        fragment_metric_d (dict): {'center': {'residues' (int): (set)},
                                   'total': {'residues' (int): (set)},
                                   'center_match_scores': {residue number(int): (list[score (float)]), ...},
                                   'match_scores': {residue number(int): (list[score (float)]), ...},
                                   'index_count': {index (int): count (int), ...}}
    Returns:
        (tuple): all_residue_score, center_residue_score
    """
    # Generate Nanohedra score for center and all residues
    all_residue_score, center_residue_score = 0, 0
    # using match scores from every residue that has been matched
    for residue_number, res_scores in fragment_metric_d['match_scores'].items():
        n = 1
        for peripheral_score in sorted(res_scores, reverse=True):
            all_residue_score += peripheral_score * (1 / float(n))
            n *= 2

    # using match scores from every central residue that has been matched
    for residue_number, res_scores in fragment_metric_d['center_match_scores'].items():
        n = 1
        for central_score in sorted(res_scores, reverse=True):
            center_residue_score += central_score * (1 / float(n))
            n *= 2

    return all_residue_score + center_residue_score, center_residue_score


def format_fragment_metrics(metrics, null=False):
    """For a set of fragment metrics, return the formatted total fragment metrics

    Returns:
        (dict): {center_residues, total_residues,
                 nanohedra_score, nanohedra_score_center, multiple_fragment_ratio, number_fragment_residues_total,
                 number_fragment_residues_center, number_fragments, percent_fragment_helix, percent_fragment_strand,
                 percent_fragment_coil}
    """
    if null:
        return fragment_metric_template
    return {'center_residues': metrics['mapped']['center']['residues'].union(metrics['paired']['center']['residues']),
            'total_residues': metrics['mapped']['total']['residues'].union(metrics['paired']['total']['residues']),
            'nanohedra_score': metrics['total']['total']['score'],
            'nanohedra_score_center': metrics['total']['center']['score'],
            'multiple_fragment_ratio': metrics['total']['multiple_ratio'],
            'number_fragment_residues_total': metrics['total']['total']['number'],
            'number_fragment_residues_center': metrics['total']['center']['number'],
            'number_fragments': metrics['total']['observations'],
            'percent_fragment_helix': (metrics['total']['index_count'][1] / (metrics['total']['observations'] * 2)),
            'percent_fragment_strand': (metrics['total']['index_count'][2] / (metrics['total']['observations'] * 2)),
            'percent_fragment_coil': ((metrics['total']['index_count'][3] + metrics['total']['index_count'][4]
                                       + metrics['total']['index_count'][5]) / (metrics['total']['observations'] * 2))}


@handle_errors(errors=KeyboardInterrupt)
def query_user_for_metrics(available_metrics, mode=None, level=None):
    """Ask the user for the desired metrics to select indices from a dataframe

    Args:
        available_metrics (set): The columns available in the DataFrame to select indices by
    Keyword Args:
        mode=None (str): The mode in which to query and format metrics information
    Returns:
        (dict)
    """
    # if mode == 'filter':
    filter_df = pd.DataFrame(master_metrics)
    direction = {'max': 'higher', 'min': 'lower'}
    instructions = {'filter': '\nFor each metric, choose values based on supported literature or design goals to '
                              'eliminate designs that are certain to fail or have sub-optimal features. Ensure your '
                              'cutoffs aren\'t too exclusive. If you end up with no designs, try relaxing your filter '
                              'values.',
                    'weight':
                        '\nFor each metric, choose a percentage signifying the metric\'s contribution to the total '
                        'selection weight. The weight will be used as a linear combination of all weights according to '
                        'each designs rank within the specified metric category. '
                        'For instance, typically the total weight should equal 1. When choosing 5 metrics, you '
                        'can assign an equal weight to each (specify 0.2 for each) or you can weight several more '
                        'strongly (0.3, 0.3, 0.2, 0.1, 0.1). When ranking occurs, for each selected metric the metric '
                        'will be sorted and designs in the top percentile will be given their percentage of the full '
                        'weight. Top percentile is defined as the most advantageous score, so the top percentile of '
                        'energy is lowest, while for hydrogen bonds it would be the most.',
                    }

    print('\n%s' % header_string % 'Select %s %s Metrics' % (level, mode))
    print('The provided dataframe will be used to select %ss based on the measured metrics from each pose. '
          'To \'%s\' designs, which metrics would you like to utilize?' % (level, mode))

    metric_values, chosen_metrics = {}, []
    end = False
    metrics_input = 'start'
    print('The available metrics are located in the top row(s) of your DataFrame. Enter your selected metrics as a '
          'comma separated input or alternatively, you can check out the available metrics by entering \'metrics\'.'
          '\nEx: \'shape_complementarity, contact_count, etc.\'')
    while not end:
        if metrics_input.lower() == 'metrics':
            print('Available Metrics\n%s\n' % ', '.join(available_metrics))
        metrics_input = input('%s' % input_string)
        chosen_metrics = set(map(str.strip, map(str.lower, metrics_input.split(','))))
        unsupported_metrics = chosen_metrics - set(available_metrics)
        if metrics_input == 'metrics':
            pass
        elif unsupported_metrics:
            print('\'%s\' not found in the DataFrame! Is your spelling correct? Have you used the correct '
                  'underscores? Please try again.' % ', '.join(unsupported_metrics))
        else:
            end = True
    correct = False
    print(instructions[mode])
    while not correct:
        for metric in chosen_metrics:
            metric_values[metric] = float(input('For \'%s\' what value of should be used for %s %sing?%s%s'
                                                % (metric, level, mode,
                                                   ' Designs with metrics %s than this value will be included'
                                                   % direction[filter_df.loc['direction', metric]].upper()
                                                   if mode == 'filter' else '', input_string)))

        print('You selected:\n\t%s' % '\n\t'.join(pretty_format_table(metric_values.items())))
        while True:
            confirm = input(confirmation_string)
            if confirm.lower() in bool_d:
                break
            else:
                print('%s %s is not a valid choice!' % (invalid_string, confirm))
        if bool_d[confirm]:
            correct = True

    return metric_values
