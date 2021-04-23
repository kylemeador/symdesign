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
from sklearn.neighbors import BallTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import PathUtils as PUtils
import SequenceProfile
# from DesignDirectory import DesignDirectory
from PDB import PDB
from Query.PDB import header_string, input_string, confirmation_string, bool_d, invalid_string
from SymDesignUtils import start_log, pickle_object, unpickle, DesignError, handle_design_errors, index_intersection, \
    remove_interior_keys, clean_dictionary, all_vs_all, condensed_to_square, sym, handle_errors, pretty_format_table, \
    digit_translate_table

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
                  # 'interface_area_hydrophobic_1_unbound':
                  #     {'description': 'Sum of each interface residue\'s hydrophobic area for interface1',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
                  # 'interface_area_hydrophobic_2_unbound':
                  #     {'description': 'Sum of each interface residue\'s hydrophobic area for interface2',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
                  # 'interface_area_polar_1_unbound':
                  #     {'description': 'Sum of each interface residue\'s polar area for interface1',
                  #      'direction': 'max', 'function': 'rank', 'filter': True},
                  # 'interface_area_polar_2_unbound':
                  #     {'description': 'Sum of each interface residue\'s polar area for interface2',
                  #      'direction': 'max', 'function': 'rank', 'filter': True},
                  # 'interface_area_total_1_unbound':
                  #     {'description': 'Sum of each interface residue\'s total area for interface1',
                  #      'direction': 'max', 'function': 'rank', 'filter': True},
                  # 'interface_area_total_2_unbound':
                  #     {'description': 'Sum of each interface residue\'s total area for interface2',
                  #      'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_area_total':
                      {'description': 'Total interface buried surface area',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_b_factor_per_residue':
                      {'description': 'The average B-factor from each atom, from each interface residue',
                       'direction': 'max', 'function': 'rank', 'filter': True},
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

nanohedra_metrics = ['nanohedra_score_per_res', 'nanohedra_score_center_per_res_center', 'nanohedra_score',
                     'nanohedra_score_center', 'number_fragment_residues_total', 'number_fragment_residues_center',
                     'multiple_fragment_ratio', 'percent_fragment_helix', 'percent_fragment_strand',
                     'percent_fragment_coil', 'number_of_fragments', 'total_interface_residues',
                     'total_non_fragment_interface_residues', 'percent_residues_fragment_total',
                     'percent_residues_fragment_center']
# These metrics are necessary for all calculations performed during the analysis script. If missing, something will fail
necessary_metrics = {'buns_complex', 'buns_1_unbound', 'buns_2_unbound', 'contact_count', 'coordinate_constraint',
                     'favor_residue_energy', 'hbonds_res_selection_complex', 'hbonds_res_selection_1_bound',
                     'hbonds_res_selection_2_bound', 'interface_connectivity_1', 'interface_connectivity_2',
                     'interface_separation', 'interface_energy_1_bound', 'interface_energy_2_bound',
                     'interface_energy_1_unbound', 'interface_energy_2_unbound', 'interface_energy_complex',
                     'interaction_energy_complex', groups, 'rosetta_reference_energy', 'rmsd', 'shape_complementarity',
                     'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
                     'sasa_hydrophobic_1_bound', 'sasa_hydrophobic_2_bound', 'sasa_polar_1_bound', 'sasa_polar_2_bound',
                     'sasa_total_1_bound', 'sasa_total_2_bound', 'solvation_energy_complex', 'solvation_energy_1_bound',
                     'solvation_energy_2_bound', 'solvation_energy_1_unbound', 'solvation_energy_2_unbound'}
#                      'buns_asu_hpol', 'buns_nano_hpol', 'buns_asu', 'buns_nano', 'buns_total',
#                      'fsp_total_stability', 'full_stability_complex',
#                      'number_hbonds', 'total_interface_residues',
#                      'average_fragment_z_score', 'nanohedra_score', 'number_of_fragments', 'interface_b_factor_per_res',

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
                     'relax_switch': groups, 'no_constraint_switch': groups, 'limit_to_profile_switch': groups,
                     'combo_profile_switch': groups, 'design_profile_switch': groups,
                     'favor_profile_switch': groups, 'consensus_design_switch': groups,
                     'interface_energy_res_summary_complex': 'interface_energy_complex',
                     'interface_energy_res_summary_1_bound': 'interface_energy_1_bound',
                     'interface_energy_res_summary_2_bound': 'interface_energy_2_bound',
                     'interface_energy_res_summary_1_unbound': 'interface_energy_1_unbound',
                     'interface_energy_res_summary_2_unbound': 'interface_energy_2_unbound',
                     'sasa_res_summary_hydrophobic_complex': 'sasa_hydrophobic_complex',
                     'sasa_res_summary_polar_complex': 'sasa_polar_complex',
                     'sasa_res_summary_total_complex': 'sasa_total_complex',
                     'sasa_res_summary_hydrophobic_1_bound': 'sasa_hydrophobic_1_bound',
                     'sasa_res_summary_polar_1_bound': 'sasa_polar_1_bound',
                     'sasa_res_summary_total_1_bound': 'sasa_total_1_bound',
                     'sasa_res_summary_hydrophobic_2_bound': 'sasa_hydrophobic_2_bound',
                     'sasa_res_summary_polar_2_bound': 'sasa_polar_2_bound',
                     'sasa_res_summary_total_2_bound': 'sasa_total_2_bound',
                     'solvation_total_energy_complex': 'solvation_energy_complex',
                     'solvation_total_energy_1_bound': 'solvation_energy_1_bound',
                     'solvation_total_energy_2_bound': 'solvation_energy_2_bound',
                     'solvation_total_energy_1_unbound': 'solvation_energy_1_unbound',
                     'solvation_total_energy_2_unbound': 'solvation_energy_2_unbound',
                     'R_int_connectivity_1': 'interface_connectivity_1',
                     'R_int_connectivity_2': 'interface_connectivity_2',
                     'ref': 'rosetta_reference_energy',
                     }
#                      'total_score': 'REU', 'decoy': 'design', 'symmetry_switch': 'symmetry',

clean_up_intermediate_columns = ['int_energy_no_intra_residue_score', 'interface_energy_bound',
                                 'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
                                 'sasa_hydrophobic_bound', 'sasa_hydrophobic_1_bound', 'sasa_hydrophobic_2_bound',
                                 'sasa_polar_bound', 'sasa_polar_1_bound', 'sasa_polar_2_bound',
                                 'sasa_total_bound', 'sasa_total_1_bound', 'sasa_total_2_bound',
                                 'buns_complex', 'buns_unbound', 'buns_1_unbound', 'buns_2_unbound',
                                 'solvation_energy_1_bound', 'solvation_energy_2_bound', 'solvation_energy_1_unbound',
                                 'solvation_energy_2_unbound',
                                 'interface_energy_1_bound', 'interface_energy_1_unbound', 'interface_energy_2_bound',
                                 'interface_energy_2_unbound',
                                 ]

# Some of these are unneeded now, but hanging around in case renaming occurred
unnecessary = ['int_area_asu_hydrophobic', 'int_area_asu_polar', 'int_area_asu_total',
               'int_area_ex_asu_hydrophobic', 'int_area_ex_asu_polar', 'int_area_ex_asu_total',
               'buns_asu', 'buns_asu_hpol', 'buns_nano', 'buns_nano_hpol',
               'int_energy_context_asu', 'int_energy_context_unbound',
               'coordinate_constraint', 'int_energy_res_summary_asu', 'int_energy_res_summary_unbound',
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
               'decoy', 'symmetry_switch', 'metrics_symmetry', 'oligomer_switch', 'total_score',
               'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex',
               'buns_asu', 'buns_asu_hpol', 'buns_nano', 'buns_nano_hpol', 'buns_total',
               ]
# remove_score_columns = ['hbonds_res_selection_asu', 'hbonds_res_selection_unbound']
#                'full_stability_oligomer_A', 'full_stability_oligomer_B']

# columns_to_remove = ['decoy', 'symmetry_switch', 'metrics_symmetry', 'oligomer_switch', 'total_score',
#                      'int_energy_context_A_oligomer', 'int_energy_context_B_oligomer', 'int_energy_context_complex']

# sum columns using tuple [0] + [1]
summation_pairs = {'buns_unbound': ('buns_1_unbound', 'buns_2_unbound'),
                   'interface_energy_bound': ('interface_energy_1_bound', 'interface_energy_2_bound'),
                   'interface_energy_unbound': ('interface_energy_1_unbound', 'interface_energy_2_unbound'),
                   'sasa_hydrophobic_bound': ('sasa_hydrophobic_1_bound', 'sasa_hydrophobic_2_bound'),
                   'sasa_polar_bound': ('sasa_polar_1_bound', 'sasa_polar_2_bound'),
                   'sasa_total_bound': ('sasa_total_1_bound', 'sasa_total_2_bound'),
                   'solvation_energy_bound': ('solvation_energy_1_bound', 'solvation_energy_2_bound'),
                   'solvation_energy_unbound': ('solvation_energy_1_unbound', 'solvation_energy_2_unbound')
                   # 'buns_hpol_total': ('buns_asu_hpol', 'buns_nano_hpol'),
                   # 'buns_heavy_total': ('buns_asu', 'buns_nano'),
                   }
# subtract columns using tuple [0] - [1] to make delta column
delta_pairs = {'interface_buried_hbonds': ('buns_complex', 'buns_unbound'),
               'interface_energy': ('interface_energy_complex', 'interface_energy_unbound'),
               # 'interface_energy_no_intra_residue_score': ('interface_energy_complex', 'interface_energy_bound'),
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
protocols_of_interest = ['design_profile', 'no_constraint']
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
    """Remove all string from dictionary keys except for string after last '_'. Ex 'design_0001' -> '0001'"""
    return {key.split('_')[-1]: pdb_dict[key] for key in pdb_dict}


def join_columns(x):
    """Combine columns in a dataframe with the same column name. Keep only the last column record"""
    new_data = ','.join(x[x.notnull()].astype(str))
    return new_data.split(',')[-1]


def columns_to_new_column(df, column_dict, mode='add'):
    """Find new column values by taking an operation of one column on another

    Args:
        df (pandas.DataFrame): Dataframe where the columns are located
        column_dict (dict[mapping[str,tuple]]): Keys are new column names, values are tuple of existing columns where
        value[0] mode(operation) value[1] = key
    Keyword Args:
        mode='add' (str) = What operator to use?
            Viable options are included in module operator, but could be 'sub', 'mul', 'truediv', etc.
    Returns:
        df (pandas.DataFrame): Dataframe with new column values
    """
    for column, pair in column_dict.items():
        try:
            df[column] = operator.attrgetter(mode)(operator)(df[pair[0]], df[pair[1]])
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
        columns (list): ['hbonds_res_selection_complex', 'hbonds_res_selection_1_unbound',
            'hbonds_res_selection_2_unbound']
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


def dirty_hbond_processing(score_dict, offset=None):  # columns
    """Process Hydrogen bond Metrics from Rosetta score dictionary

    if rosetta_numbering="true" in .xml then use offset, otherwise, hbonds are PDB numbering
    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection_complex': '15A,21A,26A,35A,...',
                            'hbonds_res_selection_1_bound': '26A'}, ...}
    Keyword Args:
        offset=None (dict): {'A': 0, 'B': 102} The amount to offset each chain by
    Returns:
        hbond_dict (dict): {'0001': [34, 54, 67, 68, 106, 178], ...}
    """
    hbond_dict = {}
    res_offset = 0
    for entry, data in score_dict.items():
        entry_dict = {}
        # for column in columns:
        for column in data:
            if column.startswith('hbonds_res_selection'):
                hbonds = data[column].split(',')
                meta_data = column.split('_')  # ['hbonds', 'res', 'selection', 'complex/interface_number', '[unbound]']
                # ensure there are hbonds present
                if hbonds[0] == '' and len(hbonds) == 1:
                    hbonds = set()
                else:
                    if meta_data[-1] == 'bound' and offset:  # 'oligomer'
                        # find offset according to chain
                        res_offset = offset[meta_data[-2]]
                    for i in range(len(hbonds)):
                        hbonds[i] = res_offset + int(hbonds[i][:-1])  # remove chain ID off last index of string
                entry_dict[meta_data[3]] = set(hbonds)
        if len(entry_dict) == 3:
            hbond_dict[entry] = list(entry_dict['complex'].difference(entry_dict['1']).difference(entry_dict['2']))
            #                                                         entry_dict['A']).difference(entry_dict['B'])
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


def interface_residue_composition_similarity(series):
    """Calculate the composition difference for pose residue classification

    Args:
        series (pandas.Series): Series with 'interface_area_total', 'core', 'rim', and 'support' indices
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


residue_template = {'energy': {'complex': 0., 'unbound': 0., 'fsp': 0., 'cst': 0.},
                    'sasa': {'polar': {'complex': 0., 'unbound': 0.}, 'hydrophobic': {'complex': 0., 'unbound': 0.},
                             'total': {'complex': 0., 'unbound': 0.}},
                    'type': None, 'hbond': 0, 'core': 0, 'interior': 0, 'rim': 0, 'support': 0}  # , 'hot_spot': 0}


def residue_processing(score_dict, mutations, columns, offset=None, hbonds=None):
    """Process Residue Metrics from Rosetta score dictionary

    Args:
        score_dict (dict): {'0001': {'buns': 2.0, 'per_res_energy_15': -3.26, ...,
                            'yhh_planarity':0.885, 'hbonds_res_selection': '15A,21A,26A,35A,...'}, ...}
        mutations (dict): {'0001': {mutation_index: {'from': 'A', 'to: 'K'}, ...}, ...}
        columns (list): ['per_res_energy_complex_5', 'per_res_sasa_polar_1_unbound_5', 
            'per_res_energy_1_unbound_5', ...]
    Keyword Args:
        offset=None (dict): {'A': 0, 'B': 102} Whether to offset the residue numbers during processing
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
                residue_dict[res] = copy.deepcopy(residue_template)
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
        offset=None (dict): {'A': 0, 'B': 102}
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
        residue_dict = {}
        # for column in columns:
        for key, value in scores.items():
            # metadata = column.split('_')
            if key.startswith('per_res_'):
                metadata = key.split('_')
                # res = int(metadata[-1])
                # res = int(metadata[-1][:-1])  # remove the chain identifier used with rosetta_numbering="False"
                res = int(metadata[-1].translate(digit_translate_table))  # remove chain_id in rosetta_numbering="False"
                if res > pose_length:
                    if not warn:  # TODO this can move to residue_processing (clean) once instated
                        warn = True
                        logger.warning('Encountered %s which has residue number > the pose length (%d). Scores above '
                                       'will be discarded. Use pbd_numbering on all Rosetta PerResidue SimpleMetrics to'
                                       ' ensure that symmetric copies have the same residue number on symmetry mates.'
                                       % (key, pose_length))
                    continue
                r_type = metadata[2]  # energy or sasa
                pose_state = metadata[-2]  # unbound or complex
                if pose_state == 'unbound' and offset:
                    res += offset[metadata[-3]]  # get oligomer chain offset
                if res not in residue_dict:
                    residue_dict[res] = copy.deepcopy(residue_template)
                if r_type == 'sasa':
                    # Ex. per_res_sasa_hydrophobic_1_unbound_15 or per_res_sasa_hydrophobic_complex_15
                    polarity = metadata[3]
                    residue_dict[res][r_type][polarity][pose_state] = value  # round(value, 3)
                    # residue_dict[res][r_type][polarity][pose_state] = round(score_dict[design][column], 3)
                else:
                    # Ex. per_res_energy_1_unbound_15 or per_res_energy_complex_15
                    residue_dict[res][r_type][pose_state] += value  # round(, 3)
        # if residue_dict:
        for res, data in residue_dict.items():
            try:
                data['type'] = mutations[design][res]  # % pose_length]
            except KeyError:
                data['type'] = mutations['reference'][res]  # % pose_length]  # fill with aa from wt seq
            if hbonds:
                if res in hbonds[design]:
                    data['hbond'] = 1
            data['energy_delta'] = data['energy']['complex'] - data['energy']['unbound']
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
            else:  # Todo remove res from dictionary as no interface design should be done? keep interior res constant?
                if relative_complex_sasa < 0.25:
                    data['interior'] = 1
                # else:
                #     residue_dict[res]['surface'] = 1
            data['coordinate_constraint'] = data['energy']['cst']
            data['residue_favored'] = data['energy']['fsp']
            data.pop('sasa')
            data.pop('energy')
            # if residue_dict[res]['energy'] <= hot_spot_energy:
            #     residue_dict[res]['hot_spot'] = 1
        # # Consolidate symmetric residues into a single design
        # for res, residue_data in residue_dict.items():
        #     if res > pose_length:
        #         new_residue_number = res % pose_length
        #         for key in residue_data:  # ['bsa_polar', 'bsa_hydrophobic', 'bsa_total', 'core', 'energy_delta', 'hbond', 'interior', 'rim', 'support', 'type']
        #             # This mechanism won't concern SASA symmetric residues info as symmetry will not be used for SASA
        #             # ex: 'bsa_polar', 'bsa_hydrophobic', 'bsa_total', 'core', 'interior', 'rim', 'support'
        #             # really only checking for energy_delta
        #             if key in ['energy_delta']:
        #                 residue_dict[new_residue_number][key] += residue_data.pop(key)

        total_residue_dict[design] = residue_dict

    return total_residue_dict


def mutation_conserved(residue_info, background):
    """Process residue mutations compared to evolutionary background. Returns 1 if residue is observed in background

    Both residue_dict and background must be same index
    Args:
        residue_info (dict): {15: {'type': 'T', ...}, ...}
        background (dict): {0: {'A': 0, 'R': 0, ...}, ...}
    Returns:
        conservation_dict (dict): {15: 1, 21: 0, 25: 1, ...}
    """
    conservation_dict = {}
    for residue, info in residue_info.items():
        residue_background = background.get(residue)
        if residue_background and residue_background[info['type']] > 0:
            conservation_dict[residue] = 1
        else:
            conservation_dict[residue] = 0

    return conservation_dict


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
        kwargs (dict): {column: {'direction': 'min', 'value': 0.3, 'idx': ['0001', '0002', ...]}, ...}
    """
    for idx in kwargs:
        if kwargs[idx]['direction'] == 'max':
            kwargs[idx]['idx'] = df[df[idx] >= kwargs[idx]['value']].index.to_list()
        if kwargs[idx]['direction'] == 'min':
            kwargs[idx]['idx'] = df[df[idx] <= kwargs[idx]['value']].index.to_list()

    return kwargs


def filter_pose(df_file, filter=None, weight=None, consensus=False):
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
        weighted_s = design_score_df.sum(axis=1).sort_values(ascending=False)
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


@handle_design_errors(errors=(DesignError, AssertionError))
def select_sequences(des_dir, weights=None, number=1, desired_protocol=None):
    """From a single design, select sequences for further characterization. If weights, then using weights the user can
     prioritize sequences, otherwise the sequence with the most neighbors will be selected

    Args:
        des_dir (DesignDirectory)
    Keyword Args:
        weights=None (iter): The weights to use in sequence selection
        number=1 (int): The number of sequences to consider for each design
        debug=False (bool): Whether or not to debug
    Returns:
        (list[tuple[DesignDirectory, str]]): Containing the selected sequences found
    """
    # Load relevant data from the design directory
    trajectory_df = pd.read_csv(des_dir.trajectories, index_col=0, header=[0])
    trajectory_df.dropna(inplace=True)
    # trajectory_df.dropna('protocol', inplace=True)
    # designs = trajectory_df.index.to_list()  # can't use with the mean and std statistics
    # designs = list(all_design_sequences[chains[0]].keys())
    logger.info('Number of starting trajectories = %d' % len(trajectory_df))

    if weights:
        filter_df = pd.DataFrame(master_metrics)
        # No filtering of protocol/indices to use as poses should have similar protocol scores coming in
        # _df = trajectory_df.loc[final_indices, :]
        _df = trajectory_df
        logger.info('Using weighting parameters: %s' % str(weights))
        _weights = {metric: {'direction': filter_df.loc['direction', metric], 'value': weights[metric]}
                    for metric in weights}
        weight_direction = {'max': True, 'min': False}  # max - ascending=False, min - ascending=True
        # weights_s = pd.Series(weights)
        weight_score_s_d = {}
        for metric in _weights:
            weight_score_s_d[metric] = _df[metric].rank(ascending=weight_direction[_weights[metric]['direction']],
                                                        method=_weights[metric]['direction'], pct=True) \
                                       * _weights[metric]['value']

        design_score_df = pd.concat([weight_score_s_d[weight] for weight in weights], axis=1)
        design_list = design_score_df.sum(axis=1).sort_values(ascending=False).index.to_list()
        logger.info('Final ranking of trajectories:\n%s' % ', '.join(pose for pose in design_list))

        return zip(repeat(des_dir), design_list[:number])
    else:
        if desired_protocol:
            unique_protocols = trajectory_df['protocol'].unique().tolist()
            while True:
                desired_protocol = input('Do you have a protocol that you prefer to pull your designs from? Possible '
                                         'protocols include:\n%s' % ', '.join(unique_protocols))
                if desired_protocol in unique_protocols:
                    break
                else:
                    print('%s is not a valid protocol, try again.' % desired_protocol)

            designs = trajectory_df[trajectory_df['protocol'] == desired_protocol].index.to_list()
        else:
            designs = trajectory_df.index.to_list()
        # sequences_pickle = glob(os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir)))
        # assert len(sequences_pickle) == 1, 'Couldn\'t find files for %s' % \
        #                                    os.path.join(des_dir.all_scores, '%s_Sequences.pkl' % str(des_dir))
        #
        # all_design_sequences = SDUtils.unpickle(sequences_pickle[0])
        # {chain: {name: sequence, ...}, ...}
        all_design_sequences = unpickle(des_dir.design_sequences)
        # all_design_sequences.pop(PUtils.stage[1])  # Remove refine from sequences, not in trajectory_df so unnecessary
        chains = list(all_design_sequences.keys())
        concatenated_sequences = [''.join([all_design_sequences[chain][design] for chain in chains])
                                  for design in designs]
        logger.debug(chains)
        logger.debug(concatenated_sequences)

        # pairwise_sequence_diff_np = SDUtils.all_vs_all(concatenated_sequences, sequence_difference)
        # Using concatenated sequences makes the values incredibly similar and inflated as most residues are the same
        # doing min/max normalization to see variation
        pairwise_sequence_diff_l = [SequenceProfile.sequence_difference(*seq_pair)
                                    for seq_pair in combinations(concatenated_sequences, 2)]
        pairwise_sequence_diff_np = np.array(pairwise_sequence_diff_l)
        _min = min(pairwise_sequence_diff_l)
        # _max = max(pairwise_sequence_diff_l)
        pairwise_sequence_diff_np = np.subtract(pairwise_sequence_diff_np, _min)
        # logger.info(pairwise_sequence_diff_l)

        # PCA analysis of distances
        pairwise_sequence_diff_mat = np.zeros((len(designs), len(designs)))
        for k, dist in enumerate(pairwise_sequence_diff_np):
            i, j = condensed_to_square(k, len(designs))
            pairwise_sequence_diff_mat[i, j] = dist
        pairwise_sequence_diff_mat = sym(pairwise_sequence_diff_mat)

        pairwise_sequence_diff_mat = StandardScaler().fit_transform(pairwise_sequence_diff_mat)
        seq_pca = PCA(PUtils.variance)
        seq_pc_np = seq_pca.fit_transform(pairwise_sequence_diff_mat)
        seq_pca_distance_vector = pdist(seq_pc_np)
        # epsilon = math.sqrt(seq_pca_distance_vector.mean()) * 0.5
        epsilon = seq_pca_distance_vector.mean() * 0.5
        logger.info('Finding maximum neighbors within distance of %f' % epsilon)

        # logger.info(pairwise_sequence_diff_np)
        # epsilon = pairwise_sequence_diff_mat.mean() * 0.5
        # epsilon = math.sqrt(seq_pc_np.myean()) * 0.5
        # epsilon = math.sqrt(pairwise_sequence_diff_np.mean()) * 0.5

        # Find the nearest neighbors for the pairwise distance matrix using the X*X^T (PCA) matrix, linear transform
        seq_neighbors = BallTree(seq_pc_np)  # Todo make brute force or automatic, not BallTree
        seq_neighbor_counts = seq_neighbors.query_radius(seq_pc_np, epsilon, count_only=True)  # , sort_results=True)
        top_count, top_idx = 0, None
        for count in seq_neighbor_counts:  # idx, enumerate()
            if count > top_count:
                top_count = count

        sorted_seqs = sorted(seq_neighbor_counts, reverse=True)
        top_neighbor_counts = sorted(set(sorted_seqs[:number]), reverse=True)

        # Find only the designs which match the top x (number) of neighbor counts
        final_designs = {designs[idx]: num_neighbors for num_neighbors in top_neighbor_counts
                         for idx, count in enumerate(seq_neighbor_counts) if count == num_neighbors}
        logger.info('The final sequence(s) and file(s):\nNeighbors\tDesign\n%s'
                    # % '\n'.join('%d %s' % (top_neighbor_counts.index(neighbors) + SDUtils.index_offset,
                    % '\n'.join('\t%d\t%s' % (neighbors, os.path.join(des_dir.designs, design))
                                for design, neighbors in final_designs.items()))

        # logger.info('Corresponding PDB file(s):\n%s' % '\n'.join('%d %s' % (i, os.path.join(des_dir.designs, seq))
        #                                                         for i, seq in enumerate(final_designs, 1)))

        # Compute the highest density cluster using DBSCAN algorithm
        # seq_cluster = DBSCAN(eps=epsilon)
        # seq_cluster.fit(pairwise_sequence_diff_np)
        #
        # seq_pc_df = pd.DataFrame(seq_pc, index=designs,
        #                       columns=['pc' + str(x + SDUtils.index_offset) for x in range(len(seq_pca.components_))])
        # seq_pc_df = pd.merge(protocol_s, seq_pc_df, left_index=True, right_index=True)

        # If final designs contains more sequences than specified, find the one with the lowest energy
        if len(final_designs) > number:
            energy_s = trajectory_df.loc[final_designs, 'interface_energy']  # includes solvation energy
            try:
                energy_s = pd.Series(energy_s)
            except ValueError:
                raise DesignError('no dataframe')
            energy_s.sort_values(inplace=True)
            final_seqs = zip(repeat(des_dir), energy_s.iloc[:number].index.to_list())
        else:
            final_seqs = zip(repeat(des_dir), final_designs.keys())

        return list(final_seqs)


@handle_design_errors(errors=(DesignError, AssertionError))
def analyze_output(des_dir, merge_residue_data=False, debug=False, save_trajectories=True, figures=True):
    """Retrieve all score information from a design directory and write results to .csv file

    Args:
        des_dir (DesignDirectory): DesignDirectory object
    Keyword Args:
        merge_residue_data (bool): Whether to incorporate residue data into Pose dataframe
        debug=False (bool): Whether to debug output
        save_trajectories=False (bool): Whether to save trajectory and residue dataframes
        figures=True (bool): Whether to make and save pose figures
    Returns:
        scores_df (pandas.DataFrame): DataFrame containing the average values from the input design directory
    """
    # Log output
    if debug:
        # global logger
        logger = start_log(name=__name__, handler=2, level=1,
                           location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    else:
        logger = start_log(name=__name__, handler=2, level=2,
                           location=os.path.join(des_dir.path, os.path.basename(des_dir.path)))
    if not des_dir.info:
        raise DesignError('Has not been initialized for design and therefore can\'t be analyzed. '
                                         'Initialize and perform interface design if you want to measure this design.')
    # TODO add fraction_buried_atoms
    # Set up pose, ensure proper input
    # global columns_to_remove, columns_to_rename, protocols_of_interest
    remove_columns = columns_to_remove
    rename_columns = columns_to_rename

    # Get design information including: interface residues, SSM's, and wild_type/design files
    profile_dict = {'combined': SequenceProfile.parse_pssm(des_dir.info['design_profile'])}
    if 'evolutionary_profile' in des_dir.info:
        profile_dict['evolution'] = SequenceProfile.parse_pssm(des_dir.info['evolutionary_profile'])
    if 'fragment_profile' in des_dir.info:
        profile_dict['fragment'] = unpickle(des_dir.info['fragment_profile'])
        issm_residues = list(set(profile_dict['fragment'].keys()))
    else:
        issm_residues = []
        logger.info('Design has no fragment information')
    interface_bkgd = SequenceProfile.get_db_aa_frequencies(PUtils.frag_directory[des_dir.info['fragment_database']])

    # Gather miscellaneous pose specific metrics
    other_pose_metrics = des_dir.pose_metrics()  # these are initialized with DesignDirectory init

    # Todo fold these into Model and attack these metrics from a Pose object
    #  This will get rid of the logger
    wild_type_file = des_dir.get_wildtype_file()
    wt_pdb = PDB.from_file(wild_type_file)
    wt_sequence = wt_pdb.atom_sequences

    int_b_factor = 0
    for residue in interface_residues:
        # if residue <= chain_sep:
        int_b_factor += wt_pdb.residue(residue).get_ave_b_factor()
    other_pose_metrics['interface_b_factor_per_res'] = round(int_b_factor / len(interface_residues), 2)

    idx_slice = pd.IndexSlice

    if not os.path.exists(des_dir.scores_file):
        other_metrics_s = pd.Series(other_pose_metrics)
        other_metrics_s = pd.concat([other_metrics_s], keys=['pose'])
        other_metrics_s = pd.concat([other_metrics_s], keys=['dock'])
    else:
        # Get the scores from all design trajectories
        all_design_scores = read_scores(os.path.join(des_dir.scores, PUtils.scores_file))
        # all_design_scores = remove_interior_keys(all_design_scores, remove_score_columns)

        # Gather mutations for residue specific processing and design sequences
        all_design_files = des_dir.get_designs()
        # all_design_files = SDUtils.get_directory_pdb_file_paths(des_dir.designs)
        # logger.debug('Design Files: %s' % ', '.join(all_design_files))
        sequence_mutations = SequenceProfile.generate_all_design_mutations(all_design_files, wild_type_file)  # TODO
        # logger.debug('Design Files: %s' % ', '.join(sequence_mutations))
        # offset_dict = AnalyzeMutatedSequences.pdb_to_pose_num(sequence_mutations['ref'])  # Removed on 01/2021 metrics.xml
        # logger.debug('Chain offset: %s' % str(offset_dict))

        # Remove wt sequence and find all designs which have corresponding pdb files
        sequence_mutations.pop('ref')
        # all_design_sequences = {AnalyzeMutatedSequences.get_pdb_sequences(file) for file in all_design_files}
        # for pdb in models:
        #     for chain in pdb.chain_id_list
        #         sequences[chain][pdb.name] = pdb.atom_sequences[chain]
        # Todo just pull from design pdbs... reorient for {chain: {name: sequence, ...}, ...} ^^
        all_design_sequences = SequenceProfile.generate_sequences(wt_sequence, sequence_mutations)
        all_design_sequences = {chain: remove_pdb_prefixes(all_design_sequences[chain]) for chain in all_design_sequences}
        all_design_scores = remove_pdb_prefixes(all_design_scores)
        logger.debug('all_design_sequences: %s' % ', '.join(name for chain in all_design_sequences
                                                            for name in all_design_sequences[chain]))
        # for chain in all_design_sequences:
        #     all_design_sequences[chain] = remove_pdb_prefixes(all_design_sequences[chain])

        # logger.debug('all_design_sequences2: %s' % ', '.join(name for chain in all_design_sequences
        #                                                      for name in all_design_sequences[chain]))
        logger.debug('all_design_scores: %s' % ', '.join(all_design_scores.keys()))
        # Ensure data is present for both scores and sequences, then initialize DataFrames
        good_designs = list(set(design for design_sequences in all_design_sequences.values() for design in design_sequences)
                            & set(all_design_scores.keys()))
        logger.info('All Designs: %s' % ', '.join(good_designs))
        all_design_scores = clean_dictionary(all_design_scores, good_designs, remove=False)
        all_design_sequences = {chain: clean_dictionary(all_design_sequences[chain], good_designs, remove=False)
                                for chain in all_design_sequences}
        logger.debug('All Sequences: %s' % all_design_sequences)

        scores_df = pd.DataFrame(all_design_scores).T
        # Gather all columns into specific types for processing and formatting TODO move up
        report_columns, per_res_columns, hbonds_columns = {}, [], []
        for column in list(scores_df.columns):
            if column.startswith('R_'):
                report_columns[column] = column.replace('R_', '')
            elif column.startswith('symmetry_switch'):
                other_pose_metrics['symmetry'] = scores_df.loc[PUtils.stage[1], column].replace('make_', '')
            elif column.startswith('per_res_'):
                per_res_columns.append(column)
            elif column.startswith('hbonds_res_selection'):
                hbonds_columns.append(column)
        rename_columns.update(report_columns)
        rename_columns.update({'R_int_sc': 'shape_complementarity', 'R_full_stability': 'full_stability_complex'})
        #                        'R_full_stability_oligomer_A': 'full_stability_oligomer_A',
        #                        'R_full_stability_oligomer_B': 'full_stability_oligomer_B',
        #                        'R_full_stability_A_oligomer': 'full_stability_oligomer_A',
        #                        'R_full_stability_B_oligomer': 'full_stability_oligomer_B',
        #                        'R_int_energy_context_A_oligomer': 'int_energy_context_oligomer_A',
        #                        'R_int_energy_context_B_oligomer': 'int_energy_context_oligomer_B'})
        #                       TODO TEST if DONE? remove the update when metrics protocol is changed
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
        scores_df = columns_to_new_column(scores_df, summation_pairs)
        scores_df = columns_to_new_column(scores_df, delta_pairs, mode='sub')
        # Remove unnecessary and Rosetta score terms except ret
        # TODO learn know how to produce them. Not in FastRelax...
        rosetta_terms_to_remove = copy.copy(rosetta_terms)
        rosetta_terms_to_remove.remove('ref')
        scores_df.drop(unnecessary + rosetta_terms_to_remove, axis=1, inplace=True, errors='ignore')

        # TODO remove dirty when columns are correct (after P432) and column tabulation precedes residue/hbond_processing
        interface_hbonds = dirty_hbond_processing(all_design_scores)  # , offset=offset_dict) when hbonds are pose numbering
        # interface_hbonds = hbond_processing(all_design_scores, hbonds_columns)  # , offset=offset_dict)

        all_mutations = SequenceProfile.generate_all_design_mutations(all_design_files, wild_type_file, pose_num=True)
        all_mutations_no_chains = SequenceProfile.make_mutations_chain_agnostic(all_mutations)
        all_mutations_simplified = SequenceProfile.simplify_mutation_dict(all_mutations_no_chains)
        cleaned_mutations = remove_pdb_prefixes(all_mutations_simplified)
        residue_dict = dirty_residue_processing(all_design_scores, cleaned_mutations, hbonds=interface_hbonds)
        #                                       offset=offset_dict)
        # can't use residue_processing (clean) in the case there is a design without metrics... columns not found!
        # residue_dict = residue_processing(all_design_scores, cleaned_mutations, per_res_columns, hbonds=interface_hbonds)
        #                                 offset=offset_dict)

        # Calculate amino acid observation percent from residue dict and background SSM's
        obs_d = {}
        for profile in profile_dict:
            obs_d[profile] = {design: mutation_conserved(residue_dict[design], profile_dict[profile])
                              for design in residue_dict}

        if 'fragment' in profile_dict:
            # Remove residues from fragment dict if no fragment information available for them
            obs_d['fragment'] = remove_interior_keys(obs_d['fragment'], issm_residues, keep=True)

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
            scores_df[r_class] = residue_df.loc[:, idx_slice[:,
                                                   residue_df.columns.get_level_values(1) == r_class]].sum(axis=1)
        scores_df['interface_composition_similarity'] = \
            scores_df.apply(interface_residue_composition_similarity, axis=1)

        interior_residue_df = residue_df.loc[:, idx_slice[:,
                                                residue_df.columns.get_level_values(1) == 'interior']].droplevel(1, axis=1)
        # Check if any of the values in columns are 1. If so, return True for that column
        interior_residues = interior_residue_df.any().index[interior_residue_df.any()].to_list()
        interface_residues = list(set(residue_df.columns.get_level_values(0).unique()) - set(interior_residues))
        assert len(interface_residues) > 0, 'No interface residues found!'
        other_pose_metrics['observations'] = len(good_designs)
        other_pose_metrics['percent_fragment'] = len(issm_residues) / len(interface_residues)
        scores_df['total_interface_residues'] = len(interface_residues)
        # 'design_residues' coming in as 234B (residue_number|chain)
        design_residues = [int(residue[:-1]) for residue in des_dir.info['design_residues'].split(',')]
        if set(interface_residues) != set(design_residues):
            logger.info('Residues %s are located in the interior' %
                        ', '.join(map(str, set(design_residues) - set(interface_residues))))

        # Interface B Factor TODO ensure asu.pdb has B-factors for Nanohedra
        # chain_sep = wt_pdb.chain(wt_pdb.chain_id_list[0]).get_terminal_residue('c').number  # only worked for 2 chains
        int_b_factor = 0
        for residue in interface_residues:
            # if residue <= chain_sep:
            int_b_factor += wt_pdb.residue(residue).get_ave_b_factor()
            # else:
            #     int_b_factor += wt_pdb.get_ave_residue_b_factor(wt_pdb.chain_id_list[1], residue)
        other_pose_metrics['interface_b_factor_per_res'] = round(int_b_factor / len(interface_residues), 2)

        pose_alignment = SequenceProfile.multi_chain_alignment(all_design_sequences)
        mutation_frequencies = clean_dictionary(pose_alignment['counts'], interface_residues, remove=False)
        # Calculate Jensen Shannon Divergence using different SSM occurrence data and design mutations
        pose_res_dict = {}
        for profile in profile_dict:  # both mut_freq and profile_dict[profile] are one-indexed
            pose_res_dict['divergence_%s' % profile] = SequenceProfile.position_specific_jsd(mutation_frequencies,
                                                                                             profile_dict[profile])
        # if 'fragment' in profile_dict:
        pose_res_dict['divergence_interface'] = SequenceProfile.compute_jsd(mutation_frequencies, interface_bkgd)
        # pose_res_dict['hydrophobic_collapse_index'] = hci()  # TODO HCI

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
            # might have to remove these from all_design_scores in the case that that is used as a dictionary again
        if residue_na_index:
            logger.warning('%s: Residue DataFrame dropped rows with missing values: %s' %
                           (des_dir.path, ', '.join(residue_na_index)))
            for res_idx in residue_na_index:
                residue_dict.pop(res_idx)
            logger.debug('Residue_dict:\n\n%s\n\n' % residue_dict)

        # Fix reported per_residue_energy to contain only interface. BUT With delta, these residues should be subtracted
        # int_residue_df = residue_df.loc[:, idx_slice[interface_residues, :]]

        # Get unique protocols for protocol specific metrics and drop unneeded protocol values
        unique_protocols = protocol_s.unique().tolist()
        for value in ['refine']:  # TODO TEST if remove '' is fixed ## after P432 MinMatch6 upon future script deployment
            try:
                unique_protocols.remove(value)
            except ValueError:
                pass
        logger.info('Unique Protocols: %s' % ', '.join(unique_protocols))

        designs_by_protocol, sequences_by_protocol = {}, {}
        stats_by_protocol = {protocol: {} for protocol in unique_protocols}
        for protocol in unique_protocols:
            designs_by_protocol[protocol] = protocol_s.index[protocol_s == protocol].tolist()
            sequences_by_protocol[protocol] = {chain: {design: all_design_sequences[chain][design]
                                                       for design in all_design_sequences[chain]
                                                       if design in designs_by_protocol[protocol]}
                                               for chain in all_design_sequences}
            protocol_alignment = SequenceProfile.multi_chain_alignment(sequences_by_protocol[protocol])
            protocol_mutation_freq = SequenceProfile.remove_non_mutations(protocol_alignment['counts'], interface_residues)
            protocol_res_dict = {'divergence_%s' % profile: SequenceProfile.position_specific_jsd(protocol_mutation_freq,
                                                                                                  profile_dict[profile])
                                 for profile in profile_dict}  # both prot_freq and profile_dict[profile] are zero indexed
            protocol_res_dict['divergence_interface'] = SequenceProfile.compute_jsd(protocol_mutation_freq,
                                                                                    interface_bkgd)

            # Get per residue divergence metric by protocol
            for key in protocol_res_dict:
                stats_by_protocol[protocol]['%s_per_res' % key] = per_res_metric(protocol_res_dict[key])  # , key=key)
                # {protocol: 'jsd_per_res': 0.747, 'int_jsd_per_res': 0.412}, ...}
            # Get per design observed background metric by protocol
            for profile in profile_dict:
                stats_by_protocol[protocol]['observed_%s' % profile] = per_res_metric(
                    {design: pose_observed_bkd[profile][design] for design in designs_by_protocol[protocol]})

            # Gather the average number of residue classifications for each protocol
            for res_class in residue_classificiation:
                stats_by_protocol[protocol][res_class] = clean_residue_df.loc[
                    designs_by_protocol[protocol],
                    idx_slice[:, clean_residue_df.columns.get_level_values(1) == res_class]].mean().sum()
            stats_by_protocol[protocol]['observations'] = len(designs_by_protocol[protocol])
        protocols_by_design = {v: k for k, _list in designs_by_protocol.items() for v in _list}

        # POSE ANALYSIS: Get total pose design statistics
        # remove below if consensus is run multiple times. the cst_weights are very large and destroy the mean
        trajectory_df = clean_scores_df.drop(PUtils.stage[5], axis=0, errors='ignore')
        assert len(trajectory_df.index.to_list()) > 0, 'No design was done on this pose'

        traj_stats = {}
        protocol_stat_df = {}
        for stat in stats_metrics:
            traj_stats[stat] = getattr(trajectory_df, stat)().rename(stat)
            protocol_stat_df[stat] = getattr(clean_scores_df.groupby(groups), stat)()
            if stat == 'mean':
                continue
            protocol_stat_df[stat].index = protocol_stat_df[stat].index.to_series().map(
                {protocol: protocol + '_' + stat for protocol in sorted(unique_protocols)})
        trajectory_df = trajectory_df.append(list(traj_stats.values()))
        # Here we add consensus back to the trajectory_df after removing above
        trajectory_df = trajectory_df.append(list(protocol_stat_df.values()))

        if merge_residue_data:
            trajectory_df = pd.merge(trajectory_df, clean_residue_df, left_index=True, right_index=True)

        # Calculate protocol significance
        # Find all unique combinations of protocols using 'mean' as all protocol combination source. Excludes Consensus
        protocol_subset_df = trajectory_df.loc[:, significance_columns]
        protocol_intersection = set(protocols_of_interest) & set(unique_protocols)

        if protocol_intersection != set(protocols_of_interest):
            logger.warning('Missing %s protocol required for significance measurements! Significance analysis failed!'
                           % ', '.join(set(protocols_of_interest) - protocol_intersection))
            significance = False
            sim_sum_and_divergence_stats, sim_measures = {}, {}
        else:
            significance = True

            sig_df = protocol_stat_df['mean']  # stats_metrics[0]
            assert len(sig_df.index.to_list()) > 1, 'Can\'t measure protocol significance, not enough protocols!'
            pvalue_df = pd.DataFrame()
            for pair in combinations(sorted(sig_df.index.to_list()), 2):
                select_df = protocol_subset_df.loc[designs_by_protocol[pair[0]] + designs_by_protocol[pair[1]], :]
                difference_s = sig_df.loc[pair[0], significance_columns].sub(
                    sig_df.loc[pair[1], significance_columns])
                pvalue_df[pair] = df_permutation_test(select_df, difference_s, group1_size=len(designs_by_protocol[pair[0]]),
                                                      compare=stats_metrics[0])
            logger.debug(pvalue_df)
            pvalue_df = pvalue_df.T  # change the significance pairs to the indices and protocol specific columns to columns
            trajectory_df = trajectory_df.append(pd.concat([pvalue_df], keys=['similarity']).swaplevel(0, 1))

            # Get pose sequence divergence TODO protocol switch
            sim_sum_and_divergence_stats = {'%s_per_res' % key: per_res_metric(pose_res_dict[key]) for key in pose_res_dict}

            # Compute sequence differences between each protocol
            residue_energy_df = \
                clean_residue_df.loc[:, idx_slice[:, clean_residue_df.columns.get_level_values(1) == 'energy_delta']]

            res_pca = PCA(PUtils.variance)  # P432 designs used 0.8 percent of the variance
            residue_energy_np = StandardScaler().fit_transform(residue_energy_df.values)
            residue_energy_pc = res_pca.fit_transform(residue_energy_np)
            residue_energy_pc_df = pd.DataFrame(residue_energy_pc, index=residue_energy_df.index,
                                                columns=['pc' + str(x + index_offset)
                                                         for x in range(len(res_pca.components_))])
            #                                    ,columns=residue_energy_df.columns)

            seq_pca = copy.deepcopy(res_pca)
            residue_dict.pop(PUtils.stage[1])  # Remove refine from analysis before PC calculation
            pairwise_sequence_diff_np = all_vs_all(residue_dict, SequenceProfile.sequence_difference)
            pairwise_sequence_diff_np = StandardScaler().fit_transform(pairwise_sequence_diff_np)
            seq_pc = seq_pca.fit_transform(pairwise_sequence_diff_np)
            # Compute the euclidean distance
            # pairwise_pca_distance_np = pdist(seq_pc)
            # pairwise_pca_distance_np = all_vs_all(seq_pc, euclidean)

            # Make PC DataFrame
            # First take all the principal components identified from above and merge with labels
            # Next the labels will be grouped and stats are taken for each group (mean is important)
            # All protocol means will have pairwise distance measured as a means of accessing similarity
            # These distance metrics will be reported in the final pose statistics
            seq_pc_df = pd.DataFrame(seq_pc, index=list(residue_dict.keys()),
                                     columns=['pc' + str(x + index_offset)
                                              for x in range(len(seq_pca.components_))])
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
                        i, j = condensed_to_square(k, len(grouped_pc_stat_df_dict[stat].index))
                        sim_measures['seq_distance'][(grouped_pc_stat_df_dict[stat].index[i],
                                                      grouped_pc_stat_df_dict[stat].index[j])] = dist

                    for k, e_dist in enumerate(energy_pca_mean_distance_vector):
                        i, j = condensed_to_square(k, len(grouped_pc_energy_df_dict[stat].index))
                        sim_measures['energy_distance'][(grouped_pc_energy_df_dict[stat].index[i],
                                                         grouped_pc_energy_df_dict[stat].index[j])] = e_dist

            for pc_stat in grouped_pc_stat_df_dict:
                logger.info(grouped_pc_stat_df_dict[pc_stat])

            # Find total protocol similarity for different metrics
            for measure in sim_measures:
                measure_s = pd.Series({pair: sim_measures[measure][pair] for pair in combinations(protocols_of_interest, 2)})
                sim_sum_and_divergence_stats['protocol_%s_sum' % measure] = measure_s.sum()

        # Create figures
        if figures:  # Todo with relevant .ipynb figures
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
            scatter = ax.scatter(seq_pc[:, 0], seq_pc[:, 1], seq_pc[:, 2], c=pc_labels_int, cmap='Spectral',
                                 edgecolor='k')
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
            scatter = ax.scatter(residue_energy_pc[:, 0], residue_energy_pc[:, 1], residue_energy_pc[:, 2],
                                 c=pc_labels_int, cmap='Spectral', edgecolor='k')
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
            pickle_object(all_design_sequences, '%s_Sequences' % str(des_dir), out_path=des_dir.all_scores)

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

        protocol_stats_s = pd.concat([pd.Series(stats_by_protocol[protocol]) for protocol in stats_by_protocol],
                                     keys=unique_protocols)

        # Add series specific Multi-index names to data
        protocol_stats_s = pd.concat([protocol_stats_s], keys=['stats'])

        if significance:
            # Find the significance between each pair of protocols
            protocol_sig_s = pd.concat([pvalue_df.loc[[pair], :].squeeze() for pair in pvalue_df.index.to_list()],
                                       keys=[tuple(pair) for pair in pvalue_df.index.to_list()])
            # squeeze turns the column headers into series indices. Keys appends to make a multi-index

            other_stats_s = pd.Series(sim_sum_and_divergence_stats)
            other_stats_s = pd.concat([other_stats_s], keys=['pose'])
            other_stats_s = pd.concat([other_stats_s], keys=['seq_design'])

            # Process similarity between protocols
            sim_measures_s = pd.concat([pd.Series(values) for values in sim_measures.values()],
                                       keys=[measure for measure in sim_measures])
            sim_series = [protocol_sig_s, other_stats_s, sim_measures_s]
        else:
            sim_series = []

        # Combine all series
        pose_s = pd.concat([pose_stat_s[stat] for stat in pose_stat_s] + [protocol_stat_s[stat] for stat in protocol_stat_s]
                           + [protocol_stats_s, other_metrics_s] + sim_series).swaplevel(0, 1)

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
        logger = start_log(name='main', level=1)
        logger.debug('Debug mode. Verbose output')
    else:
        logger = start_log(name='main', level=2)

    logger.info('Starting %s with options:\n%s' %
                (os.path.basename(__file__),
                 '\n'.join([str(arg) + ':' + str(getattr(args, arg)) for arg in vars(args)])))

    # Collect all designs to be processed
    all_poses, location = collect_designs(file=args.file, directory=args.directory)
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
        mp_threads = calculate_mp_threads()
        logger.info('Starting multiprocessing using %s threads' % str(mp_threads))
        zipped_args = zip(all_design_directories, repeat(args.delta_g), repeat(args.join), repeat(args.debug),
                          repeat(save))
        pose_results, exceptions = SDUtils.mp_starmap(analyze_output, zipped_args, mp_threads)
    else:
        logger.info('Starting processing. If single process is taking awhile, use -m during submission')
        pose_results, exceptions = [], []
        for des_directory in all_design_directories:
            result, error = analyze_output(des_directory, delta_refine=args.delta_g, merge_residue_data=args.join,
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
