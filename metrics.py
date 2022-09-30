from __future__ import annotations

import math
import operator
import warnings
from copy import copy
from itertools import repeat
from json import loads
from typing import Literal, AnyStr, Any, Sequence, Iterable

import numpy as np
import pandas as pd

import structure
from structure.utils import protein_letters_literal, alphabet_types, create_translation_tables
from utils.path import groups, reference_name, structure_background, design_profile, hbnet_design_profile
from resources.query.utils import input_string, validate_type, verify_choice, header_string
from utils import handle_errors, start_log, pretty_format_table, index_intersection, digit_translate_table, \
    DesignError, z_score

# Globals
logger = start_log(name=__name__)
_min, _max = 'min', 'max'
rank, normalize, boolean = 'rank', 'normalize', 'boolean'
metric_weight_functions = [rank, normalize]
residue_classificiation = ['core', 'rim', 'support']  # 'hot_spot'
per_residue_energy_states = ['complex', 'bound', 'unbound', 'solv_complex', 'solv_bound', 'solv_unbound']
energy_metric_names = ['interface_energy_complex', 'interface_energy_bound', 'interface_energy_unbound',
                       'interface_solvation_energy_complex', 'interface_solvation_energy_bound',
                       'interface_solvation_energy_unbound']
energy_metrics_rename_mapping = dict(zip(per_residue_energy_states, energy_metric_names))
errat_1_sigma, errat_2_sigma, errat_3_sigma = 5.76, 11.52, 17.28  # These are approximate magnitude of deviation
collapse_significance_threshold = 0.43
collapse_reported_std = .05
idx_slice = pd.IndexSlice
master_metrics = {
    'average_fragment_z_score':
        dict(description='The average fragment z-value used in docking/design',
             direction=_min, function=normalize, filter=True),
    # 'buns_heavy_total':
    #     dict(description='Buried unsaturated H-bonding heavy atoms in the design',
    #      direction=_min, function=rank, filter=True),
    # 'buns_hpol_total':
    #     dict(description='Buried unsaturated H-bonding polarized hydrogen atoms in the design',
    #      direction=_min, function=rank, filter=True),
    # 'buns_total':
    #     dict(description='Total buried unsaturated H-bonds in the design',
    #      direction=_min, function=rank, filter=True),
    'buried_unsatisfied_hbond_density':
        dict(description='Buried Unsaturated Hbonds per Angstrom^2 of interface',
             direction=_min, function=normalize, filter=True),
    'buried_unsatisfied_hbonds':
        dict(description='Total buried unsatisfied H-bonds in the design',
             direction=_min, function=rank, filter=True),
    # 'component_1_symmetry':
    #     dict(description='The symmetry group of component 1',
    #      direction=_min, 'function': 'equals', 'filter': True),
    # 'component_1_name':
    #     dict(description='component 1 PDB_ID', direction=None, function=None, filter=False),
    # 'component_1_number_of_residues':
    #     dict(description='The number of residues in the monomer of component 1',
    #      direction=_max, function=rank, filter=True),
    # 'component_1_max_radius':
    #     dict(description='The maximum distance that component 1 reaches away from the center of mass',
    #      direction=_max, function=normalize, filter=True),
    # 'component_1_n_terminal_helix':
    #     dict(description='Whether the n-terminus has an alpha helix',
    #      direction=None, function=None, filter=True),  # Todo binary?
    # 'component_1_c_terminal_helix':
    #     dict(description='Whether the c-terminus has an alpha helix',
    #      direction=None, function=None, filter=True),  # Todo binary?
    # 'component_1_n_terminal_orientation':
    #     dict(description='The direction the n-terminus is oriented from the symmetry group center of mass.'
    #                     ' 1 is away, -1 is towards', direction=None, function=None, filter=False),
    # 'component_1_c_terminal_orientation':
    #     dict(description='The direction the c-terminus is oriented from the symmetry group center of mass.'
    #                     ' 1 is away, -1 is towards', direction=None, function=None, filter=False),
    # 'component_2_symmetry':
    #     dict(description='The symmetry group of component 2',
    #      direction=_min, 'function': 'equals', 'filter': True),
    # 'component_2_name':
    #     dict(description='component 2 PDB_ID', direction=None, function=None, filter=False),
    # 'component_2_number_of_residues':
    #     dict(description='The number of residues in the monomer of component 2',
    #      direction=_min, function=rank, filter=True),
    # 'component_2_max_radius':
    #     dict(description='The maximum distance that component 2 reaches away from the center of mass',
    #      direction=_max, function=normalize, filter=True),
    # 'component_2_n_terminal_helix':
    #     dict(description='Whether the n-terminus has an alpha helix',
    #      direction=None, function=None, filter=True),  # Todo binary?
    # 'component_2_c_terminal_helix':
    #     dict(description='Whether the c-terminus has an alpha helix',
    #      direction=None, function=None, filter=True),  # Todo binary?
    # 'component_2_n_terminal_orientation':
    #     dict(description='The direction the n-terminus is oriented from the symmetry group center of mass.'
    #                     ' 1 is away, -1 is towards', direction=None, function=None, filter=False),
    # 'component_2_c_terminal_orientation':
    #     dict(description='The direction the c-terminus is oriented from the symmetry group center of mass.'
    #                     ' 1 is away, -1 is towards', direction=None, function=None, filter=False),
    'contact_count':
        dict(description='Number of carbon-carbon contacts across interface',
             direction=_max, function=rank, filter=True),
    'contact_order_collapse_significance':
        dict(description='Summed significance values taking product of positive collapse and contact order per residue.'
                         ' Positive values indicate collapse in areas with low contact order. Negative, collapse in '
                         'high contact order. A protein fold relying on high contact order may not need as much '
                         'collapse, while without high contact order, the segment should rely on itself to fold',
             direction=_max, function=rank, filter=True),
    'contact_order_collapse_z_sum':
        dict(description='Summed contact order z score, scaled proportionally by positions with increased collapse. '
                         'More negative is more isolated collapse. Positive indicates collapse is occurring in '
                         'predominantly higher contact order sites',
             direction=_min, function=rank, filter=True),
    'core':
        dict(description='The number of "core" residues as classified by E. Levy 2010',
             direction=_max, function=rank, filter=True),
    'coordinate_constraint':
        dict(description='Total weight of coordinate constraints to keep design from moving in cartesian '
                         'space', direction=_min, function=normalize, filter=True),
    'design_dimension':
        dict(description='The underlying dimension of the design. 0 - point, 2 - layer, 3 - space group',
             direction=_min, function=normalize, filter=True),
    'divergence_design_per_residue':
        dict(description='The Jensen-Shannon divergence of interface residues from the position specific '
                         'design profile values. Includes fragment & evolution if both are True, otherwise '
                         'only includes those specified for use in design.',
             direction=_min, function=rank, filter=True),
    'divergence_fragment_per_residue':
        dict(description='The Jensen-Shannon divergence of interface residues from the position specific '
                         'fragment profile',
             direction=_min, function=rank, filter=True),
    'divergence_evolution_per_residue':
        dict(description='The Jensen-Shannon divergence of interface residues from the position specific '
                         'evolutionary profile',
             direction=_min, function=rank, filter=True),
    'divergence_interface_per_residue':
        dict(description='The Jensen-Shannon divergence of interface residues from the typical interface '
                         'background',
             direction=_min, function=rank, filter=True),
    'energy_distance_from_structure_background_mean':
        dict(description='The distance of the design\'s per residue energy from a design with no constraint on'
                         ' amino acid selection',
             direction=_min, function=rank, filter=True),
    'entity_1_c_terminal_helix':  # TODO make a single metric
        dict(description='Whether the entity has a c-terminal helix',
             direction=_max, function=boolean, filter=True),
    'entity_1_c_terminal_orientation':  # TODO make a single metric
        dict(description='Whether the entity c-termini is closer to the assembly core or surface (1 is away, -1 is '
                         'towards',
             direction=_max, function=rank, filter=True),
    'entity_1_max_radius':  # TODO make a single metric
        dict(description='The furthest point the entity reaches from the assembly core',
             direction=_min, function=rank, filter=True),
    'entity_1_min_radius':  # TODO make a single metric
        dict(description='The closest point the entity approaches the assembly core',
             direction=_max, function=rank, filter=True),
    'entity_1_n_terminal_helix':  # TODO make a single metric
        dict(description='Whether the entity has a n-terminal helix',
             direction=_max, function=boolean, filter=True),
    'entity_1_n_terminal_orientation':  # TODO make a single metric
        dict(description='Whether the entity n-termini is closer to the assembly core or surface (1 is away, -1 is '
                         'towards)',
             direction=_max, function=rank, filter=True),
    'entity_1_name':  # TODO make a single metric
        dict(description='The name of the entity',
             direction=None, function=None, filter=None),
    'entity_1_number_of_mutations':  # TODO make a single metric
        dict(description='The number of mutations made',
             direction=_min, function=rank, filter=True),
    'entity_1_number_of_residues':  # TODO make a single metric
        dict(description='The number of residues',
             direction=_min, function=rank, filter=True),
    'entity_1_percent_mutations':  # TODO make a single metric
        dict(description='The percentage of the entity that has been mutated',
             direction=_min, function=rank, filter=True),
    'entity_1_radius':  # TODO make a single metric
        dict(description='The center of mass of the entity from the assembly core',
             direction=_min, function=rank, filter=True),
    'entity_1_symmetry':  # TODO make a single metric
        dict(description='The symmetry notation of the entity',
             direction=None, function=None, filter=None),
    'entity_1_thermophile':  # TODO make a single metric
        dict(description='Whether the entity is a thermophile',
             direction=_max, function=boolean, filter=None),
    'entity_max_radius_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the max radii of each entity '
                         'from one another', direction=_min, function=rank, filter=True),
    'entity_max_radius_ratio_1v2':  # TODO make a single metric
        dict(description='The ratio of the maximum radius from a reference of component 1 versus 2',
             direction=None, function=None, filter=None),
    'entity_maximum_radius':
        dict(description='The maximum radius any entity extends from the assembly core',
             direction=_min, function=rank, filter=True),
    'entity_min_radius_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the min radii of each entity from'
                         ' one another',
             direction=_min, function=rank, filter=True),
    'entity_min_radius_ratio_1v2':  # TODO make a single metric
        dict(description='The ratio of the minimum radius from a reference of component 1 versus 2',
             direction=None, function=None, filter=None),
    'entity_minimum_radius':
        dict(description='The minimum radius any entity approaches the assembly core',
             direction=_max, function=rank, filter=True),
    'entity_number_of_residues_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the number of residues of each '
                         'entity from one another',
             direction=_min, function=rank, filter=True),
    'entity_number_of_residues_ratio_1v2':  # TODO make a single metric
        dict(description='', direction=None, function=None, filter=None),
    'entity_radius_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the center of mass of each entity'
                         ' from one another',
             direction=_min, function=rank, filter=True),
    'entity_radius_ratio_1v2':  # TODO make a single metric
        dict(description='', direction=None, function=None, filter=None),
    'entity_residue_length_total':
        dict(description='The total number of residues in the design',
             direction=_min, function=rank, filter=True),
    'entity_thermophilicity':
        dict(description='The extent to which the entities in the pose are thermophilic',
             direction=_max, function=rank, filter=True),
    'errat_accuracy':
        dict(description='The overall Errat score of the design',
             direction=_max, function=rank, filter=True),
    'errat_deviation':
        dict(description='Whether a residue window deviates significantly from typical Errat distribution',
             direction=_min, function=boolean, filter=True),
    'favor_residue_energy':
        dict(description='Total weight of sequence constraints used to favor certain amino acids in design. '
                         'Only protocols with a favored profile have values',
             direction=_max, function=normalize, filter=True),
    # 'fragment_z_score_total':
    #     dict(description='The sum of all fragments z-values',
    #      direction=None, function=None, filter=None),
    'interaction_energy_complex':
        dict(description='The two-body (residue-pair) energy of the complexed interface. No solvation '
                         'energies', direction=_min, function=rank, filter=True),
    'global_collapse_z_sum':
        dict(description='The sum of all sequence regions z-scores experiencing increased collapse. Measures the '
                         'normalized magnitude of additional hydrophobic collapse',
             direction=_min, function=rank, filter=True),
    'hydrophobicity_deviation_magnitude':
        dict(description='The total deviation in the hydrophobic collapse, either more or less collapse '
                         'prone', direction=_min, function=rank, filter=True),
    'interface_area_hydrophobic':
        dict(description='Total hydrophobic interface buried surface area',
             direction=_min, function=rank, filter=True),
    'interface_area_polar':
        dict(description='Total polar interface buried surface area',
             direction=_max, function=rank, filter=True),
    'interface_area_to_residue_surface_ratio':
        dict(description='The average ratio of interface buried surface area to the surface accessible residue'
                         ' area in the uncomplexed pose',
             direction=_max, function=rank, filter=True),
    'interface_area_total':
        dict(description='Total interface buried surface area',
             direction=_max, function=rank, filter=True),
    'interface_b_factor_per_residue':
        dict(description='The average B-factor from each atom, from each interface residue',
             direction=_max, function=rank, filter=True),
    'interface_bound_activation_energy':
        dict(description='Energy required for the unbound interface to adopt the conformation in the '
                         'complexed state', direction=_min, function=rank, filter=True),
    'interface_composition_similarity':
        dict(description='The similarity to the expected interface composition given interface buried surface '
                         'area. 1 is similar to natural interfaces, 0 is dissimilar',
             direction=_max, function=rank, filter=True),
    'interface_connectivity_1':  # TODO make a single metric
        dict(description='How embedded is interface1 in the rest of the protein?',
             direction=_max, function=normalize, filter=True),
    # 'interface_connectivity_2':
    #     dict(description='How embedded is interface2 in the rest of the protein?',
    #      direction=_max, function=normalize, filter=True),
    'interface_connectivity':
        dict(description='How embedded is the total interface in the rest of the protein?',
             direction=_max, function=normalize, filter=True),
    # 'int_energy_density':
    #     dict(description='Energy in the bound complex per Angstrom^2 of interface area',
    #      direction=_min, function=rank, filter=True),
    'interface_energy':
        dict(description='DeltaG of the complexed and unbound (repacked) interfaces',
             direction=_min, function=rank, filter=True),
    'interface_energy_complex':
        dict(description='Total interface residue energy summed in the complexed state',
             direction=_min, function=rank, filter=True),
    'interface_energy_density':
        dict(description='Interface energy per interface area^2. How much energy is achieved within the '
                         'given space?', direction=_min, function=rank, filter=True),
    'interface_energy_unbound':
        dict(description='Total interface residue energy summed in the unbound state',
             direction=_min, function=rank, filter=True),
    'interface_energy_1_unbound':  # TODO make a single metric or remove
        dict(description='Sum of interface1 residue energy in the unbound state',
             direction=_min, function=rank, filter=True),
    # 'interface_energy_2_unbound':
    #     dict(description='Sum of interface2 residue energy in the unbound state',
    #      direction=_min, function=rank, filter=True),
    'interface_separation':
        dict(description='Median distance between all atom points on each side of the interface',
             direction=_min, function=normalize, filter=True),
    'interface_separation_core':
        dict(description='Median distance between all atom points on each side of the interface core fragment '
                         'positions',
             direction=_min, function=normalize, filter=True),
    'interface_secondary_structure_count':
        dict(description='The number of unique secondary structures in the interface',
             direction=_max, function=normalize, filter=True),
    'interface_secondary_structure_fragment_count':
        dict(description='The number of unique fragment containing secondary structures in the interface',
             direction=_max, function=normalize, filter=True),
    # DSSP G:310 helix, H:α helix and I:π helix, B:beta bridge, E:strand/beta bulge, T:turns,
    #      S:high curvature (where the angle between i-2, i, and i+2 is at least 70°), and " "(space):loop
    'interface_secondary_structure_fragment_topology':
        dict(description='The Stride based secondary structure names of each unique element where possible '
                         'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                         'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
             direction=None, function=None, filter=None),
    'interface_secondary_structure_fragment_topology_1':  # TODO make a single metric or remove
        dict(description='The Stride based secondary structure names of each unique element where possible '
                         'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                         'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
             direction=None, function=None, filter=None),
    'interface_secondary_structure_topology':
        dict(description='The Stride based secondary structure names of each unique element where possible '
                         'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                         'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
             direction=None, function=None, filter=None),
    'interface_secondary_structure_topology_1':  # TODO make a single metric or remove
        dict(description='The Stride based secondary structure names of each unique element where possible '
                         'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                         'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
             direction=None, function=None, filter=None),
    'interface_local_density':
        dict(description='A measure of the average number of atom neighbors for each atom in the interface',
             direction=_max, function=rank, filter=True),
    'multiple_fragment_ratio':
        dict(description='The extent to which fragment observations are connected in the interface. Higher '
                         'ratio means multiple fragment observations per residue',
             direction=_max, function=rank, filter=True),
    'nanohedra_score':
        dict(description='Sum of total fragment containing residue match scores (1 / 1 + Z-score^2) weighted '
                         'by their ranked match score. Maximum of 2/residue',
             direction=_max, function=rank, filter=True),
    'nanohedra_score_center':
        dict(description='nanohedra_score for the central fragment residues only',
             direction=_max, function=rank, filter=True),
    'nanohedra_score_center_normalized':
        dict(description='The central Nanohedra Score normalized by number of central fragment residues',
             direction=_max, function=rank, filter=True),
    'nanohedra_score_normalized':
        dict(description='The Nanohedra Score normalized by number of fragment residues',
             direction=_max, function=rank, filter=True),
    'new_collapse_island_significance':
        dict(description='The magnitude of the contact_order_collapse_significance (abs(deviation)) for identified '
                         'new collapse islands',
             direction=_min, function=rank, filter=True),
    'new_collapse_islands':
        dict(description='The number of new collapse islands found',
             direction=_min, function=rank, filter=True),
    'number_fragment_residues_total':
        dict(description='The number of residues in the interface with fragment observationsfound',
             direction=_max, function=rank, filter=True),
    'number_fragment_residues_center':
        dict(description='The number of interface residues that belong to a central fragment residue',
             direction=_max, function=rank, filter=None),
    'number_hbonds':
        dict(description='The number of residues making H-bonds in the total interface. Residues may make '
                         'more than one H-bond', direction=_max, function=rank, filter=True),
    'number_of_fragments':
        dict(description='The number of fragments found in the pose interface',
             direction=_max, function=normalize, filter=True),
    'number_of_mutations':
        dict(description='The number of mutations made to the pose (ie. wild-type residue to any other '
                         'amino acid)', direction=_min, function=normalize, filter=True),
    'observations':
        dict(description='Number of unique design trajectories contributing to statistics',
             direction=_max, function=rank, filter=True),
    'observed_design':
        dict(description='Percent of observed residues in combined profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_evolution':
        dict(description='Percent of observed residues in evolutionary profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_fragment':
        dict(description='Percent of observed residues in the fragment profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_interface':
        dict(description='Percent of observed residues in fragment profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'percent_core':
        dict(description='The percentage of residues which are "core" according to Levy, E. 2010',
             direction=_max, function=normalize, filter=True),
    'percent_fragment':
        dict(description='Percent of residues with fragment data out of total residues',
             direction=_max, function=normalize, filter=True),
    'percent_fragment_coil':
        dict(description='The percentage of fragments represented from coiled SS elements',
             direction=_max, function=normalize, filter=True),
    'percent_fragment_helix':
        dict(description='The percentage of fragments represented from an a-helix SS elements',
             direction=_max, function=normalize, filter=True),
    'percent_fragment_strand':
        dict(description='The percentage of fragments represented from a b-strand SS elements',
             direction=_max, function=normalize, filter=True),
    'percent_interface_area_hydrophobic':
        dict(description='The percent of interface area which is occupied by hydrophobic atoms',
             direction=_min, function=normalize, filter=True),
    'percent_interface_area_polar':
        dict(description='The percent of interface area which is occupied by polar atoms',
             direction=_min, function=normalize, filter=True),
    'percent_mutations':
        dict(description='The percent of the design which has been mutated',
             direction=_max, function=normalize, filter=True),
    'percent_residues_fragment_center':
        dict(description='The percentage of residues which are central fragment observations',
             direction=_max, function=normalize, filter=True),
    'percent_residues_fragment_total':
        dict(description='The percentage of residues which are represented by fragment observations',
             direction=_max, function=normalize, filter=True),
    'percent_rim':
        dict(description='The percentage of residues which are "rim" according to Levy, E. 2010',
             direction=_min, function=normalize, filter=True),
    'percent_support':
        dict(description='The percentage of residues which are "support" according to Levy, E. 2010',
             direction=_max, function=normalize, filter=True),
    groups:
        dict(description='Protocols utilized to search sequence space given fragment and/or evolutionary '
                         'constraint information',
             direction=None, function=None, filter=False),
    'protocol_energy_distance_sum':
        dict(description='The distance between the average linearly embedded per residue energy co-variation '
                         'between specified protocols. Larger = greater distance. A small distance indicates '
                         'that different protocols arrived at the same per residue energy conclusions despite '
                         'different pools of amino acids specified for sampling',
             direction=_min, function=rank, filter=True),
    'protocol_similarity_sum':
        dict(description='The statistical similarity between all sampled protocols. Larger is more similar, '
                         'indicating that different protocols have interface statistics that are similar '
                         'despite different pools of amino acids specified for sampling',
             direction=_max, function=rank, filter=True),
    'protocol_sequence_distance_sum':
        dict(description='The distance between the average linearly embedded sequence differences between '
                         'specified protocols. Larger = greater distance. A small distance indicates that '
                         'different protocols arrived at the same per residue energy conclusions despite '
                         'different pools of amino acids specified for sampling',
             direction=_min, function=rank, filter=True),
    'rim':
        dict(description='The number of "rim" residues as classified by E. Levy 2010',
             direction=_max, function=rank, filter=True),
    # 'rmsd':
    #     dict(description='Root Mean Square Deviation of all CA atoms between the refined (relaxed) and '
    #                     'designed states', direction=_min, function=normalize, filter=True),
    'rmsd_complex':
        dict(description='Root Mean Square Deviation of all CA atoms between the refined (relaxed) and '
                         'designed states',
             direction=_min, function=normalize, filter=True),
    'rosetta_reference_energy':
        dict(description='Rosetta Energy Term - A metric for the unfolded energy of the protein along with '
                         'sequence fitting corrections',
             direction=_max, function=rank, filter=True),
    'sequential_collapse_peaks_z_sum':
        dict(description='The collapse z-score for each residue scaled sequentially by the number of '
                         'previously observed collapsable locations',
             direction=_max, function=rank, filter=True),
    'sequential_collapse_z_sum':
        dict(description='The collapse z-score for each residue scaled by the proximity to sequence start',
             direction=_max, function=rank, filter=True),
    'shape_complementarity':
        dict(description='Measure of fit between two surfaces from Lawrence and Colman 1993',
             direction=_max, function=normalize, filter=True),
    'shape_complementarity_core':
        dict(description='Measure of fit between two surfaces from Lawrence and Colman 1993 at interface core '
                         'fragment positions',
             direction=_max, function=normalize, filter=True),
    'interface_solvation_energy':  # free_energy of desolvation is positive for bound interfaces. unbound - complex
        dict(description='The free energy resulting from hydration of the separated interface surfaces. '
                         'Positive values indicate poorly soluble surfaces upon dissociation',
             direction=_min, function=rank, filter=True),
    'interface_solvation_energy_activation':  # unbound - bound
        dict(description='The free energy of solvation resulting from packing the bound, uncomplexed state to '
                         'an unbound, uncomplexed state. Positive values indicate a tendency towards the bound'
                         ' configuration',
             direction=_min, function=rank, filter=True),
    'interface_solvation_energy_bound':
        dict(description='The desolvation free energy of the separated interface surfaces. Positive values '
                         'indicate energy is required to desolvate',
             direction=_min, function=rank, filter=True),
    'interface_solvation_energy_complex':
        dict(description='The desolvation free energy of the complexed interface. Positive values indicate '
                         'energy is required to desolvate',
             direction=_min, function=rank, filter=True),
    'interface_solvation_energy_unbound':
        dict(description='The desolvation free energy of the separated, repacked, interface surfaces. Positive'
                         ' values indicate energy is required to desolvate',
             direction=_min, function=rank, filter=True),
    'support':
        dict(description='The number of "support" residues as classified by E. Levy 2010',
             direction=_max, function=rank, filter=True),
    # 'symmetry':
    #     dict(description='The specific symmetry type used design (point (0), layer (2), lattice(3))',
    #      direction=None, function=None, filter=False),
    'symmetry_group_1':  # TODO make a single metric
        dict(description='The specific symmetry of group 1 from of Nanohedra symmetry combination materials '
                         '(SCM)',
             direction=None, function=None, filter=False),
    # 'symmetry_group_2':
    #     dict(description='The specific symmetry of group 1 from of Nanohedra symmetry combination '
    #                     'materials (SCM)',
    #      direction=None, function=None, filter=False),
    'total_interface_residues':
        dict(description='The total number of interface residues found in the pose (residue CB within 8A)',
             direction=_max, function=rank, filter=True),
    'total_non_fragment_interface_residues':
        dict(description='The number of interface residues that are missing central fragment observations',
             direction='max', function=rank, filter=True),
    'REU':
        dict(description='Rosetta Energy Units. Always 0. We can disregard',
             direction=_min, function=rank, filter=True),
    'time':
        dict(description='Time for the protocol to complete',
             direction=None, function=None, filter=None),
    'hbonds_res_selection_unbound':
        dict(description='The specific h-bonds present in the bound pose',
             direction=None, function=None, filter=None),
    'hbonds_res_selection_1_unbound':
        dict(description='The specific h-bonds present in the unbound interface1',
             direction=None, function=None, filter=None),
    'hbonds_res_selection_2_unbound':
        dict(description='The specific h-bonds present in the unbound interface2',
             direction=None, function=None, filter=None),
    'dslf_fa13':
        dict(description='Rosetta Energy Term - disulfide bonding',
             direction=None, function=None, filter=None),
    'fa_atr':
        dict(description='Rosetta Energy Term - lennard jones full atom atractive forces',
             direction=None, function=None, filter=None),
    'fa_dun':
        dict(description='Rosetta Energy Term - Dunbrack rotamer library statistical probability',
             direction=None, function=None, filter=None),
    'fa_elec':
        dict(description='Rosetta Energy Term - full atom electrostatic forces',
             direction=None, function=None, filter=None),
    'fa_intra_rep':
        dict(description='Rosetta Energy Term - lennard jones full atom intra-residue repulsive forces',
             direction=None, function=None, filter=None),
    'fa_intra_sol_xover4':
        dict(description='Rosetta Energy Term - full atom intra-residue solvent forces',
             direction=None, function=None, filter=None),
    'fa_rep':
        dict(description='Rosetta Energy Term - lennard jones full atom repulsive forces',
             direction=None, function=None, filter=None),
    'fa_sol':
        dict(description='Rosetta Energy Term - full atom solvent forces',
             direction=None, function=None, filter=None),
    'hbond_bb_sc':
        dict(description='Rosetta Energy Term - backbone/sidechain hydrogen bonding',
             direction=None, function=None, filter=None),
    'hbond_lr_bb':
        dict(description='Rosetta Energy Term - long range backbone hydrogen bonding',
             direction=None, function=None, filter=None),
    'hbond_sc':
        dict(description='Rosetta Energy Term - side-chain hydrogen bonding',
             direction=None, function=None, filter=None),
    'hbond_sr_bb':
        dict(description='Rosetta Energy Term - short range backbone hydrogen bonding',
             direction=None, function=None, filter=None),
    'lk_ball_wtd':
        dict(description='Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
             direction=None, function=None, filter=None),
    'omega':
        dict(description='Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
             direction=None, function=None, filter=None),
    'p_aa_pp':
        dict(description='"Rosetta Energy Term - statistical probability of an amino acid given angles phi',
             direction=None, function=None, filter=None),
    'pro_close':
        dict(description='Rosetta Energy Term - to favor closing of proline rings',
             direction=None, function=None, filter=None),
    'rama_prepro':
        dict(description='Rosetta Energy Term - amino acid dependent term to favor certain Ramachandran angles'
                         ' on residue before proline',
             direction=None, function=None, filter=None),
    'yhh_planarity':
        dict(description='Rosetta Energy Term - favor planarity of tyrosine alcohol hydrogen',
             direction=None, function=None, filter=None)
}
filter_df = pd.DataFrame(master_metrics)
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
                     'interface_separation',
                     # 'interface_energy_1_bound', 'interface_energy_1_unbound',  'interface_energy_complex',
                     'interaction_energy_complex', groups, 'rosetta_reference_energy', 'shape_complementarity',
                     # 'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
                     # 'sasa_hydrophobic_1_bound', 'sasa_polar_1_bound', 'sasa_total_1_bound',
                     # 'solvation_energy_complex', 'solvation_energy_1_bound', 'solvation_energy_1_unbound'
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
rosetta_required_metrics = []
# unused, just a placeholder for the metrics in population
final_metrics = {'buried_unsatisfied_hbonds', 'contact_count', 'core', 'coordinate_constraint',
                 'divergence_design_per_residue', 'divergence_evolution_per_residue', 'divergence_fragment_per_residue',
                 'divergence_interface_per_residue', 'entity_thermophilicity', 'favor_residue_energy',
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
                 'percent_interface_area_hydrophobic', 'percent_interface_area_polar', 'percent_mutations',
                 'percent_residues_fragment_center',
                 'percent_residues_fragment_total', 'percent_rim', 'percent_support',
                 'protocol_energy_distance_sum', 'protocol_similarity_sum', 'protocol_seq_distance_sum',
                 'rosetta_reference_energy', 'rim', 'rmsd', 'shape_complementarity', 'interface_solvation_energy',
                 'interface_solvation_energy_activation', 'support',
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
                     'shape_complementarity_core_median_dist': 'interface_core_separation',
                     'ref': 'rosetta_reference_energy',
                     'interaction_energy_density_filter': 'interaction_energy_per_residue'
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
# Todo clean up these columns for master branch...
clean_up_intermediate_columns = ['int_energy_no_intra_residue_score',  # 'interface_energy_bound', Todo make _3_..., _4_
                                 'sasa_hydrophobic_complex', 'sasa_polar_complex', 'sasa_total_complex',
                                 'sasa_hydrophobic_bound', 'sasa_hydrophobic_1_bound', 'sasa_hydrophobic_2_bound',
                                 'sasa_polar_bound', 'sasa_polar_1_bound', 'sasa_polar_2_bound',
                                 'sasa_total_bound', 'sasa_total_1_bound', 'sasa_total_2_bound',
                                 'buns_complex', 'buns_unbound', 'buns_1_unbound', 'buns_2_unbound',
                                 'solvation_energy', 'solvation_energy_complex',
                                 'solvation_energy_1_bound', 'solvation_energy_2_bound', 'solvation_energy_1_unbound',
                                 'solvation_energy_2_unbound',
                                 'interface_energy_1_bound', 'interface_energy_1_unbound', 'interface_energy_2_bound',
                                 'interface_energy_2_unbound',
                                 'interface_solvation_energy_bound', 'interface_solvation_energy_bound',
                                 'interface_solvation_energy_unbound', 'interface_solvation_energy_complex'
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

# subtract columns using tuple [0] - [1] to make delta column
delta_pairs = {'buried_unsatisfied_hbonds': ('buns_complex', 'buns_unbound'),  # Rosetta
               'interface_energy': ('interface_energy_complex', 'interface_energy_unbound'),  # Rosetta
               # 'interface_energy_no_intra_residue_score': ('interface_energy_complex', 'interface_energy_bound'),
               'interface_bound_activation_energy': ('interface_energy_bound', 'interface_energy_unbound'),  # Rosetta
               'interface_solvation_energy': ('interface_solvation_energy_unbound', 'interface_solvation_energy_complex'),  # Rosetta
               'interface_solvation_energy_activation': ('interface_solvation_energy_unbound', 'interface_solvation_energy_bound'),  # Rosetta
               # 'interface_area_hydrophobic': ('sasa_hydrophobic_bound', 'sasa_hydrophobic_complex'),
               # 'interface_area_polar': ('sasa_polar_bound', 'sasa_polar_complex'),
               # 'interface_area_total': ('sasa_total_bound', 'sasa_total_complex')
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
                  'buried_unsatisfied_hbond_density': ('buried_unsatisfied_hbonds', 'interface_area_total'),  # Rosetta
                  'interface_energy_density': ('interface_energy', 'interface_area_total')}  # Rosetta

# All Rosetta based score terms ref is most useful to keep for whole pose to give "unfolded ground state"
rosetta_terms = ['lk_ball_wtd', 'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'yhh_planarity', 'dslf_fa13',
                 'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                 'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'ref']

# Current protocols in use in interface_design.xml
protocols = ['design_profile_switch', 'favor_profile_switch', 'limit_to_profile_switch', 'structure_background_switch']
protocols_of_interest = {design_profile, structure_background, hbnet_design_profile}  # Todo adapt to any user protocol!
# protocols_of_interest = ['combo_profile', 'limit_to_profile', 'no_constraint']  # Used for P432 models

protocol_column_types = ['mean', 'sequence_design']  # 'stats',
# Specific columns of interest to distinguish between design trajectories
significance_columns = ['buried_unsatisfied_hbonds',
                        'contact_count', 'interface_energy', 'interface_area_total', 'number_hbonds',
                        'percent_interface_area_hydrophobic', 'shape_complementarity', 'interface_solvation_energy']
# sequence_columns = ['divergence_evolution_per_residue', 'divergence_fragment_per_residue',
#                     'observed_evolution', 'observed_fragment']
multiple_sequence_alignment_dependent_metrics = \
    ['global_collapse_z_sum', 'hydrophobicity_deviation_magnitude', 'new_collapse_island_significance',
     'new_collapse_islands', 'sequential_collapse_peaks_z_sum', 'sequential_collapse_z_sum']
profile_dependent_metrics = ['divergence_evolution_per_residue', 'observed_evolution']
frag_profile_dependent_metrics = ['divergence_fragment_per_residue', 'observed_fragment']
# per_res_keys = ['jsd', 'des_jsd', 'int_jsd', 'frag_jsd']


def read_scores(file: AnyStr, key: str = 'decoy') -> dict[str, dict[str, str]]:
    """Take a json formatted metrics file and incorporate entries into nested dictionaries with "key" as outer key

    Automatically formats scores according to conventional metric naming scheme, ex: "R_", "S_", or "M_" prefix removal

    Args:
        file: Location on disk of scorefile
        key: Name of the json key to use as outer dictionary identifier
    Returns:
        {design_name: {metric_key: metric_value, ...}, ...}
    """
    with open(file, 'r') as f:
        scores = {}
        for json_entry in f.readlines():
            # entry = loads(json_entry)
            formatted_scores = {}
            for score, value in loads(json_entry).items():
                if score.startswith('per_res_'):  # there are a lot of these scores in particular
                    formatted_scores[score] = value
                elif score.startswith('R_'):
                    formatted_scores[score.replace('R_', '').replace('S_', '')] = value
                else:
                    # res_summary replace is used to take sasa_res_summary and other per_res metrics "string" off
                    score = score.replace('res_summary_', '').replace('solvation_total', 'solvation')
                    formatted_scores[columns_to_rename.get(score, score)] = value

            design = formatted_scores.pop(key)
            if design not in scores:
                scores[design] = formatted_scores
            else:
                # # to ensure old trajectories don't have lingering protocol info
                # for protocol in protocols:
                #     if protocol in entry:  # ensure that the new scores has a protocol before removing the old one.
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


def columns_to_new_column(df, column_dict, mode='add'):
    """Set new column value by taking an operation of one column on another

    Can perform summation and subtraction if a set of columns is provided
    Args:
        df (pandas.DataFrame): Dataframe where the columns are located
        column_dict (dict[mapping[str,tuple]]): Keys are new column names, values are tuple of existing columns where
            df[key] = value[0] mode(operation) value[1]
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
        except IndexError:
            raise IndexError(f'Tthe number of columns in the set {column_set} is not >= 2. {new_column} not possible!')
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
            parsed_hbonds = set(int(hbond.translate(digit_translate_table))
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
            if not column.startswith('hbonds_res_selection'):
                continue
            meta_data = column.split('_')  # ['hbonds', 'res', 'selection', 'complex/interface_number', '[unbound]']
            parsed_hbonds = set(int(hbond.translate(digit_translate_table))
                                for hbond in value.split(',') if hbond != '')  # check if '' in case no hbonds
            # if meta_data[-1] == 'bound' and offset:  # find offset according to chain
            #     res_offset = offset[meta_data[-2]]
            #     parsed_hbonds = set(residue + res_offset for residue in parsed_hbonds)
            if meta_data[3] == 'complex':
                complex_bonds = parsed_hbonds
            else:  # from another state
                unbound_bonds = unbound_bonds.union(parsed_hbonds)
        if complex_bonds:  # 'complex', '1', '2'
            hbonds[design] = complex_bonds.difference(unbound_bonds)
            # hbonds[design] = [hbonds_entry['complex'].difference(hbonds_entry['1']).difference(hbonds_entry['2']))]
            #                                                         hbonds_entry['A']).difference(hbonds_entry['B'])
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


def interface_composition_similarity(series):
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


def incorporate_mutation_info(design_residue_scores: dict,
                              mutations: dict[str, 'structure.sequence.mutation_dictionary']) -> dict:
    """Incorporate mutation measurements into residue info. design_residue_scores and mutations must be the same index

    Args:
        design_residue_scores: {'001': {15: {'complex': -2.71, 'bound': [-1.9, 0], 'unbound': [-1.9, 0],
                                             'solv_complex': -2.71, 'solv_bound': [-1.9, 0], 'solv_unbound': [-1.9, 0],
                                             'fsp': 0., 'cst': 0.}, ...}, ...}
        mutations: {'reference': {mutation_index: {'from': 'A', 'to: 'K'}, ...},
                    '001': {mutation_index: {}, ...}, ...}
    Returns:
        {'001': {15: {'type': 'T', 'energy_delta': -2.71, 'coordinate_constraint': 0. 'residue_favored': 0., 'hbond': 0}
                 ...}, ...}
    """
    warn = False
    reference_data = mutations.get(reference_name)
    pose_length = len(reference_data)
    for design, residue_info in design_residue_scores.items():
        mutation_data = mutations.get(design)
        if not mutation_data:
            continue

        remove_residues = []
        for residue_number, data in residue_info.items():
            try:  # Set residue AA type based on provided mutations
                data['type'] = mutation_data[residue_number]
            except KeyError:  # Residue is not in mutations, probably missing as it is not a mutation
                try:  # Fill in with AA from reference_name seq
                    data['type'] = reference_data[residue_number]
                except KeyError:  # Residue is out of bounds on pose length
                    # Possibly a virtual residue or string that was processed incorrectly from the digit_translate_table
                    if not warn:
                        logger.error(f'Encountered residue number "{residue_number}" which is not within the pose size '
                                     f'"{pose_length}" and will be removed from processing. This is likely an error '
                                     f'with residue processing or residue selection in the specified rosetta protocol.'
                                     f' If there were warnings produced indicating a larger residue number than pose '
                                     f'size, this problem was not addressable heuristically and something else has '
                                     f'occurred. It is likely that this residue number is not useful if you indeed have'
                                     f' output_as_pdb_nums="true"')
                        warn = True
                    remove_residues.append(residue_number)
                    continue

        # Clean up any incorrect residues
        for residue in remove_residues:
            residue_info.pop(residue)

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


def calculate_collapse_metrics(sequences_of_interest: Iterable[Iterable[Sequence[str]]],
                               # poses_of_interest: list['structure.model.Pose'],
                               residue_contact_order_z: np.ndarray, reference_collapse: np.ndarray,
                               collapse_profile: np.ndarray = None) -> list[dict[str, float]]:
    """Measure folding metrics from sequences based on reference per residue contact order and hydrophobic collapse
    parameters

    Args:
        sequences_of_interest:
        residue_contact_order_z:
        reference_collapse:
        collapse_profile:
    Returns:
        The collapse metric dictionary (metric, value pairs) for each concatenated sequence in the provided
            sequences_of_interest
    """
    # The contact order is always positive. Negating makes it inverted as to weight more highly contacting poorly
    residue_contact_order_inverted_z = residue_contact_order_z * -1
    # reference_collapse = hydrophobic_collapse_index(self.api_db.sequences.retrieve_data(name=entity.name))
    reference_collapse_bool = np.where(reference_collapse > collapse_significance_threshold, 1, 0)
    # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]

    if collapse_profile is not None:
        collapse_profile_mean = np.nanmean(collapse_profile, axis=-2)
        collapse_profile_std = np.nanstd(collapse_profile, axis=-2)
        # Use only the reference (index=0) hydrophobic_collapse_index to calculate a reference
        reference_collapse_z_score = z_score(collapse_profile[0], collapse_profile_mean, collapse_profile_std)

    # A measure of the sequential, the local, the global, and the significance all constitute interesting
    # parameters which contribute to the outcome. I can use the measure of each to do a post-hoc solubility
    # analysis. In the meantime, I could stay away from any design which causes the global collapse to increase
    # by some percent of total relating to the z-score. This could also be an absolute which would tend to favor
    # smaller proteins. Favor smaller or larger? What is the literature/data say about collapse?
    #
    # A synopsis of my reading is as follows:
    # I hypothesize that the worst offenders in collapse modification will be those that increase in
    # hydrophobicity in sections intended for high contact order packing. Additionally, the establishment of new
    # collapse locales will be detrimental to the folding pathway regardless of their location, however
    # establishment in folding locations before a significant protein core is established are particularly
    # egregious. If there is already collapse occurring, the addition of new collapse could be less important as
    # the structural determinants (geometric satisfaction) of the collapse are not as significant
    #
    # All possible important aspects measured are:
    # X the sequential collapse (earlier is worse than later as nucleation of core is wrong),
    #   sequential_collapse_peaks_z_sum, sequential_collapse_z_sum
    # X the local nature of collapse (is the sequence/structural context amenable to collapse?),
    #   contact_order_collapse_z_sum
    # X the global nature of collapse (how much has collapse increased globally),
    #   hydrophobicity_deviation_magnitude, global_collapse_z_sum,
    # X the change from "non-collapsing" to "collapsing" where collapse passes a threshold and changes folding
    #   new_collapse_islands, new_collapse_island_significance

    # linearly weight residue by sequence position (early > late) with the halfway position (midpoint)
    # weighted at 1
    midpoint = .5
    scale = 1 / midpoint
    folding_and_collapse = []
    # for pose_idx, pose in enumerate(poses_of_interest):
    #     standardized_collapse = []
    #     for entity_idx, entity in enumerate(pose.entities):
    #         sequence_length = entity.number_of_residues
    #         standardized_collapse.append(entity.hydrophobic_collapse)
    for pose_idx, sequences in enumerate(sequences_of_interest):
        # Gather all the collapse info for the particular sequence group
        standardized_collapse = [hydrophobic_collapse_index(sequence) for entity_idx, sequence in enumerate(sequences)]
        # Todo
        #  Calculate two HCI ?at the same time? to benchmark the two hydrophobicity scales
        #   -> observed_collapse, standardized_collapse = hydrophobic_collapse_index(sequence)
        standardized_collapse = np.concatenate(standardized_collapse)
        sequence_length = standardized_collapse.shape[0]
        # find collapse where: delta above standard collapse, collapsable boolean, and successive number
        # collapse_propensity = np.where(standardized_collapse > 0.43, standardized_collapse - 0.43, 0)
        # scale the collapse propensity by the standard collapse threshold and make z score
        collapse_propensity_z = z_score(standardized_collapse, collapse_significance_threshold, collapse_reported_std)
        positive_collapse_propensity_z = np.maximum(collapse_propensity_z, 0)
        # ^ [0, 0, 0, 0, 0.04, 0.06, 0, 0, 0.1, 0.07, ...]

        collapse_bool = positive_collapse_propensity_z != 0  # [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, ...]
        # collapse_bool = np.where(positive_collapse_propensity_z, 1, 0)
        increased_collapse = np.where(collapse_bool - reference_collapse_bool == 1, 1, 0)
        # Check if increased collapse positions resulted in a location of new collapse.
        # i.e. sites where a new collapse is formed compared to wild-type
        # Ex, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ...]
        # new_collapse = np.zeros_like(collapse_bool)
        # list is faster to index than np.ndarray so we use here
        new_collapse = [True if _bool and (not reference_collapse[idx - 1] or not reference_collapse[idx + 1])
                        else False
                        for idx, _bool in enumerate(increased_collapse[1:-1].tolist(), 1)]

        new_collapse_peak_start = [0 for _ in range(collapse_bool.shape[0])]  # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, ...]
        collapse_peak_start = copy(new_collapse_peak_start)  # [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, ...]
        sequential_collapse_points = np.zeros_like(collapse_bool)  # [0, 0, 0, 0, 1, 1, 0, 0, 2, 2, ..]
        collapse_iterator = 0
        for prior_idx, idx in enumerate(range(1, collapse_propensity_z.shape[0])):
            # Check for the new_collapse "islands" and collapse_peak_start index by comparing neighboring residues
            # Both conditions are only True when 0 -> 1 transition occurs
            if new_collapse[prior_idx] < new_collapse[idx]:
                new_collapse_peak_start[idx] = 1
            if collapse_bool[prior_idx] < collapse_bool[idx]:
                collapse_peak_start[idx] = 1
                collapse_iterator += 1
            sequential_collapse_points[idx] = collapse_iterator

        if collapse_profile is not None:
            # Compare the measured collapse to the metrics gathered from the collapse_profile
            z_array = z_score(standardized_collapse, collapse_profile_mean, collapse_profile_std)
            # Find indices where the z_array is increased/decreased compared to the reference_collapse_z_score
            # Todo
            #  Test for magnitude and directory of the wt versus profile.
            #  Remove subtraction? It seems useful...
            difference_collapse_z = z_array - reference_collapse_z_score
            # Find the indices where the sequence collapse has increased compared to reference collapse_profile
            global_collapse_z = np.maximum(difference_collapse_z, 0)
            # Sum the contact order, scaled proportionally by the collapse increase. More negative is more isolated
            # collapse. Positive indicates poor maintaning of the starting collapse
            contact_order_collapse_z_sum = np.sum(residue_contact_order_z * global_collapse_z)
            # The sum of all sequence regions z-scores experiencing increased collapse. Measures the normalized
            # magnitude of additional hydrophobic collapse
            global_collapse_z_sum = global_collapse_z.sum()
            hydrophobicity_deviation_magnitude = np.abs(difference_collapse_z).sum()

            # Reduce sequential_collapse_points iter to only points where collapse_bool is True (1)
            sequential_collapse_points *= collapse_bool
            step = 1 / sum(collapse_peak_start)  # This is 1 over the "total_collapse_points"
            add_step_array = collapse_bool * step
            # v [0, 0, 0, 0, 2, 2, 0, 0, 1.8, 1.8, ...]
            sequential_collapse_weights = scale * ((1 - step * sequential_collapse_points) + add_step_array)
            sequential_collapse_peaks_z_sum = np.sum(sequential_collapse_weights * global_collapse_z)
            # v [2, 1.98, 1.96, 1.94, 1.92, ...]
            sequential_weights = scale * (1 - np.arange(sequence_length) / sequence_length)
            sequential_collapse_z_sum = np.sum(sequential_weights * global_collapse_z)
        else:
            hydrophobicity_deviation_magnitude, contact_order_collapse_z_sum, sequential_collapse_peaks_z_sum, \
                sequential_collapse_z_sum, global_collapse_z_sum = 0., 0., 0., 0., 0.

        # With 'new_collapse_island_significance'
        #  Use contact order and hci to understand designability of an area and its folding modification
        #  Indicate the degree to which low contact order segments (+) are reliant on collapse for folding, while
        #  high contact order (-) use collapse

        #  For positions experiencing collapse, multiply by inverted contact order
        collapse_significance = residue_contact_order_inverted_z * positive_collapse_propensity_z
        #  Positive values indicate collapse in areas with low contact order
        #  Negative, collapse in high contact order

        # Add the concatenated collapse metrics to total
        folding_and_collapse.append({'new_collapse_islands': sum(new_collapse_peak_start),
                                     'new_collapse_island_significance': np.sum(new_collapse_peak_start
                                                                                * np.abs(collapse_significance)),
                                     'contact_order_collapse_significance': collapse_significance,
                                     'contact_order_collapse_z_sum': contact_order_collapse_z_sum,
                                     'global_collapse_z_sum': global_collapse_z_sum,
                                     'hydrophobicity_deviation_magnitude': hydrophobicity_deviation_magnitude,
                                     'sequential_collapse_peaks_z_sum': sequential_collapse_peaks_z_sum,
                                     'sequential_collapse_z_sum': sequential_collapse_z_sum})

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
    complex_hydro = per_residue_df.loc[:, idx_slice[:, 'sasa_hydrophobic_complex']]
    bsa_hydrophobic = (bound_hydro.rename(columns={'sasa_hydrophobic_bound': 'bsa_hydrophobic'})
                       - complex_hydro.rename(columns={'sasa_hydrophobic_complex': 'bsa_hydrophobic'}))

    bound_polar = per_residue_df.loc[:, idx_slice[:, 'sasa_polar_bound']]
    complex_polar = per_residue_df.loc[:, idx_slice[:, 'sasa_polar_complex']]
    bsa_polar = (bound_polar.rename(columns={'sasa_polar_bound': 'bsa_polar'})
                 - complex_polar.rename(columns={'sasa_polar_complex': 'bsa_polar'}))

    # Make sasa_complex_total columns
    bound_total = (bound_hydro.rename(columns={'sasa_hydrophobic_bound': 'sasa_total_bound'})
                   + bound_polar.rename(columns={'sasa_polar_bound': 'sasa_total_bound'}))
    complex_total = (complex_hydro.rename(columns={'sasa_hydrophobic_complex': 'sasa_total_complex'})
                     + complex_polar.rename(columns={'sasa_polar_complex': 'sasa_total_complex'}))

    bsa_total = (bsa_hydrophobic.rename(columns={'bsa_hydrophobic': 'bsa_total'})
                 + bsa_polar.rename(columns={'bsa_polar': 'bsa_total'}))

    # Find the relative sasa of the complex and the unbound fraction
    buried_interface_residues = (bsa_total > 0).to_numpy()
    # ^ support, rim or core
    # surface_or_rim = per_residue_df.loc[:, idx_slice[index_residues, 'sasa_relative_complex']] > 0.25
    core_or_interior = per_residue_df.loc[:, idx_slice[:, 'sasa_relative_complex']] < 0.25
    surface_or_rim = ~core_or_interior
    support_not_core = per_residue_df.loc[:, idx_slice[:, 'sasa_relative_bound']] < 0.25
    # core_sufficient = np.logical_and(core_or_interior, buried_interface_residues).to_numpy()
    core_residues = np.logical_and(~support_not_core,
                                   (np.logical_and(core_or_interior, buried_interface_residues)).to_numpy()).rename(
        columns={'sasa_relative_bound': 'core'})
    interior_residues = np.logical_and(core_or_interior, ~buried_interface_residues).rename(
        columns={'sasa_relative_complex': 'interior'})
    support_residues = np.logical_and(support_not_core, buried_interface_residues).rename(
        columns={'sasa_relative_bound': 'support'})
    rim_residues = np.logical_and(surface_or_rim, buried_interface_residues).rename(
        columns={'sasa_relative_complex': 'rim'})
    surface_residues = np.logical_and(surface_or_rim, ~buried_interface_residues).rename(
        columns={'sasa_relative_complex': 'surface'})

    per_residue_df = per_residue_df.join([bsa_hydrophobic, bsa_polar, bsa_total, bound_total, complex_total,
                                          core_residues, interior_residues, support_residues, rim_residues,
                                          surface_residues
                                          ])
    # per_residue_df = pd.concat([per_residue_df, core_residues, interior_residues, support_residues, rim_residues,
    #                             surface_residues], axis=1)
    return per_residue_df


def sum_per_residue_metrics(per_residue_df: pd.DataFrame) -> pd.DataFrame:
    """From a DataFrame with per residue values, tabulate the values relating to interfacial energy and solvation energy

    Args:
        per_residue_df: The DataFrame with MultiIndex columns where level1=residue_numbers, level0=residue_metric
    Returns:
        A new DataFrame with the summation of all residue_numbers in the per_residue columns
    """
    summed_energies = \
        {energy_state: per_residue_df.loc[:, idx_slice[:, energy_state]].sum(axis=1)
         for energy_state in per_residue_energy_states}
    summed_residue_classification = \
        {residue_class: per_residue_df.loc[:, idx_slice[:, residue_class]].sum(axis=1)
         for residue_class in residue_classificiation}

    summed_scores_df = pd.DataFrame({**summed_energies, **summed_residue_classification})

    return summed_scores_df.rename(columns=energy_metrics_rename_mapping)


def calculate_sequence_observations_and_divergence(alignment: 'structure.sequence.MultipleSequenceAlignment',
                                                   backgrounds: dict[str, np.ndarray],
                                                   select_indices: list[int] = None) \
        -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
                  # position_specific_jsd(pose_alignment.frequencies, background)[interface_indexer]
                  position_specific_divergence(alignment.frequencies, background)[select_indices]
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
#             with warnings.catch_warnings() as w:
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
# divergence of p from q
# Dkl = SUMi->N(probability(pi) * log(probability(qi))
# This is also the Shannon Entropy...

def kl_divergence(frequencies: np.ndarray, bgd_frequencies: np.ndarray) -> float:
    """Calculate Kullback–Leibler Divergence value from observed and background frequencies

    Args:
        frequencies: [0.05, 0.001, 0.1, ...]
        bgd_frequencies: [0, 0, ...]
    Returns:
        Bounded between 0 and 1. 1 is more divergent from background frequencies
    """
    probs1 = bgd_frequencies * np.log(frequencies)
    return np.where(np.isnan(probs1), 0, probs1).sum()


def js_divergence(frequencies: np.ndarray, bgd_frequencies: np.ndarray, lambda_: float = 0.5) -> float:
    """Calculate Jensen-Shannon Divergence value from observed and background frequencies

    Args:
        frequencies: [0.05, 0.001, 0.1, ...]
        bgd_frequencies: [0, 0, ...]
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
    with warnings.catch_warnings() as w:
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
#             with warnings.catch_warnings() as w:
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


def filter_df_for_index_by_value(df: pd.DataFrame, metrics: dict) -> dict:
    """Retrieve the indices from a DataFrame which have column values greater_equal/less_equal to an indicated threshold

    Args:
        df: DataFrame to filter indices on
        metrics: {column: 0.3, ...} OR {column: {'direction': 'min', 'value': 0.3}, ...} to specify a sorting direction
    Returns:
        {column: ['0001', '0002', ...], ...}
    """
    filtered_indices = {}
    for column, value in metrics.items():
        if isinstance(value, dict):
            specification = value.get('direction')  # Todo make an ability to use boolean?
            # todo convert specification options 'greater' '>' 'greater than' to 'max'/'min'
            value = value.get('value', 0.)
        else:
            specification = filter_df.loc['direction', column]

        if specification == _max:
            filtered_indices[column] = df[df[column] >= value].index.to_list()
        elif specification == _min:
            filtered_indices[column] = df[df[column] <= value].index.to_list()

    return filtered_indices


def prioritize_design_indices(df: pd.DataFrame | AnyStr, filter: dict | bool = None,
                              weight: dict | bool = None, protocol: str | list[str] = None, **kwargs) -> pd.DataFrame:
    """Return a filtered/sorted DataFrame (both optional) with indices that pass a set of filters and/or are ranked
    according to a feature importance. Both filter and weight instructions are provided or queried from the user

    Caution: Expects that provided DataFrame is of particular formatting, i.e. 3 column MultiIndices, 1 index indices.
    If the DF varies from this, this function will likely cause errors
    Args:
        df: DataFrame to filter/weight indices
        filter: Whether to remove viable candidates by certain metric values or a mapping of value and filter threshold
            pairs
        weight: Whether to rank the designs by metric values or a mapping of value and weight pairs where the total
            weight will be the sum of all individual weights
        protocol: Whether specific design protocol(s) should be chosen
    Returns:
        The dataframe of selected designs based on the provided filters and weights. DataFrame contains MultiIndex
            columns with size 3
    """
    idx_slice = pd.IndexSlice
    # Grab pose info from the DateFrame and drop all classifiers in top two rows.
    if isinstance(df, pd.DataFrame):
        if list(range(3 - df.columns.nlevels)):
            df = pd.concat([df], axis=1, keys=[tuple('pose' for _ in range(3 - df.columns.nlevels))])
    else:
        df = pd.read_csv(df, index_col=0, header=[0, 1, 2])
        df.replace({False: 0, True: 1, 'False': 0, 'True': 1}, inplace=True)
    logger.info('Number of starting designs: %d' % len(df))

    if protocol:
        if isinstance(protocol, str):  # treat protocol as a str
            protocol = [protocol]

        try:
            protocol_df = df.loc[:, idx_slice[protocol, protocol_column_types, :]]
        except KeyError:
            logger.info('Protocol "%s" was not found in the set of designs...' % protocol)
            # raise DesignError('The protocol "%s" was not found in the set of designs...')
            available_protocols = df.columns.get_level_values(0).unique()
            while True:
                protocol = input('What protocol would you like to choose?%s\nAvailable options are: %s%s'
                                 % (describe_string, ', '.join(available_protocols), input_string))
                if protocol in available_protocols:
                    protocol = [protocol]  # todo make multiple protocols available for input ^
                    break
                elif protocol in describe:
                    describe_data(df=df)
                else:
                    print('Invalid protocol %s. Please choose one of %s' % (protocol, ', '.join(available_protocols)))
            protocol_df = df.loc[:, idx_slice[protocol, protocol_column_types, :]]
        protocol_df.dropna(how='all', inplace=True, axis=0)  # drop completely empty rows in case of groupby ops
        # ensure 'dock'ing data is present in all protocols
        simple_df = pd.merge(df.loc[:, idx_slice[['pose'], ['dock'], :]], protocol_df, left_index=True, right_index=True)
        logger.info('Number of designs after protocol selection: %d' % len(simple_df))
    else:
        protocol = ['pose']  # Todo change to :?
        simple_df = df.loc[:, idx_slice[protocol, df.columns.get_level_values(1) != 'std', :]]
    # this is required for a multi-index column where the different protocols are in the top row of the df columns
    simple_df = pd.concat([simple_df.loc[:, idx_slice[prot, :, :]].droplevel(0, axis=1).droplevel(0, axis=1)
                           for prot in protocol])
    simple_df.dropna(how='all', inplace=True, axis=0)
    # simple_df = simple_df.droplevel(0, axis=1).droplevel(0, axis=1)  # simplify headers

    if filter:
        if isinstance(filter, dict):
            filters = filter
        else:
            available_filters = simple_df.columns.to_list()
            filters = query_user_for_metrics(available_filters, df=simple_df, mode='filter', level='design')
        logger.info('Using filter parameters: %s' % str(filters))

        # When df is not ranked by percentage
        # _filters = {metric: {'direction': filter_df.loc['direction', metric], 'value': value}
        #             for metric, value in filters.items()}

        # Filter the DataFrame to include only those values which are le/ge the specified filter
        filtered_indices = filter_df_for_index_by_value(simple_df, filters)  # **_filters)
        # filtered_indices = {metric: filters_with_idx[metric]['idx'] for metric in filters_with_idx}
        logger.info('Number of designs passing filters:\n\t%s'
                    % '\n\t'.join('%6d - %s' % (len(indices), metric) for metric, indices in filtered_indices.items()))
        final_indices = index_intersection(filtered_indices.values())
        logger.info('Number of designs passing all filters: %d' % len(final_indices))
        if len(final_indices) == 0:
            raise DesignError('There are no poses left after filtering! Try choosing less stringent values or make '
                              'better designs!')
        simple_df = simple_df.loc[final_indices, :]

    # {column: {'direction': _min, 'value': 0.3, 'idx_slice': ['0001', '0002', ...]}, ...}
    if weight:
        if isinstance(weight, dict):
            weights = weight
        else:
            available_metrics = simple_df.columns.to_list()
            weights = query_user_for_metrics(available_metrics, df=simple_df, mode='weight', level='design')
        logger.info('Using weighting parameters: %s' % str(weights))
        design_ranking_s = rank_dataframe_by_metric_weights(simple_df, weights=weights, **kwargs)
        design_ranking_s.name = 'selection_weight'
        final_df = pd.merge(design_ranking_s, simple_df, left_index=True, right_index=True)
        final_df = pd.concat([final_df], keys=[('pose', 'metric')], axis=1)
        # simple_df = pd.concat([simple_df], keys=df.columns.levels[0:1])
        # weighted_df = pd.concat([design_ranking_s], keys=[('-'.join(weights), 'sum', 'selection_weight')], axis=1)
        # final_df = pd.merge(weighted_df, simple_df, left_index=True, right_index=True)
        # final_df = pd.merge(weighted_df, df, left_index=True, right_index=True)
    else:
        final_df = simple_df.loc[simple_df.sort_values('interface_energy', ascending=True).index, :]

    # final_df is sorted by the best value to the worst
    return final_df


describe_string = '\nTo see a describtion of the data, enter "describe"\n'
describe = ['describe', 'desc', 'DESCRIBE', 'DESC', 'Describe', 'Desc']


def describe_data(df=None):
    """Describe the DataFrame to STDOUT"""
    print('The available metrics are located in the top row(s) of your DataFrame. Enter your selected metrics as a '
          'comma separated input. To see descriptions for only certain metrics, enter them here. '
          'Otherwise, hit "Enter"')
    metrics_input = input('%s' % input_string)
    chosen_metrics = set(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                            repeat(' '), repeat('_'))))
    # metrics = input('To see descriptions for only certain metrics, enter them here. Otherwise, hit "Enter"%s'
    #                 % input_string)
    if not chosen_metrics:
        columns_of_interest = slice(None)
        pass
    else:
        columns_of_interest = [idx for idx, column in enumerate(df.columns.get_level_values(-1).to_list())
                               if column in chosen_metrics]
    # format rows/columns for data display, then revert
    max_columns, min_columns = pd.get_option('display.max_columns'), pd.get_option('display.max_rows')
    pd.set_option('display.max_columns', None), pd.set_option('display.max_rows', None)
    print(df.iloc[:, columns_of_interest].describe())
    pd.set_option('display.max_columns', max_columns), pd.set_option('display.max_rows', min_columns)


@handle_errors(errors=KeyboardInterrupt)
def query_user_for_metrics(available_metrics, df=None, mode=None, level=None):
    """Ask the user for the desired metrics to select indices from a dataframe

    Args:
        available_metrics (Iterable): The columns available in the DataFrame to select indices by
    Keyword Args:
        mode=None (str): The mode in which to query and format metrics information
    Returns:
        (dict)
    """
    try:
        direction = dict(max='higher', min='lower')
        instructions = \
            {'filter': '\nFor each metric, choose values based on supported literature or design goals to eliminate '
                       'designs that are certain to fail or have sub-optimal features. Ensure your cutoffs aren\'t too '
                       'exclusive. If you end up with no designs, try relaxing your filter values.',
             'weight':
                 '\nFor each metric, choose a percentage signifying the metric\'s contribution to the total selection '
                 'weight. The weight will be used as a linear combination of all weights according to each designs rank'
                 ' within the specified metric category. For instance, typically the total weight should equal 1. When '
                 'choosing 5 metrics, you can assign an equal weight to each (specify 0.2 for each) or you can weight '
                 'several more strongly (0.3, 0.3, 0.2, 0.1, 0.1). When ranking occurs, for each selected metric the '
                 'metric will be sorted and designs in the top percentile will be given their percentage of the full '
                 'weight. Top percentile is defined as the most advantageous score, so the top percentile of energy is '
                 'lowest, while for hydrogen bonds it would be the most.'}

        print('\n%s' % header_string % 'Select %s %s Metrics' % (level, mode))
        print('The provided dataframe will be used to select %ss based on the measured metrics from each pose. '
              'To "%s" designs, which metrics would you like to utilize?%s'
              % (level, mode, describe_string if df is not None else ''))

        print('The available metrics are located in the top row(s) of your DataFrame. Enter your selected metrics as a '
              'comma separated input or alternatively, you can check out the available metrics by entering "metrics".'
              '\nEx: "shape_complementarity, contact_count, etc."')
        metrics_input = input('%s' % input_string)
        chosen_metrics = set(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                                repeat(' '), repeat('_'))))
        while True:  # unsupported_metrics or 'metrics' in chosen_metrics:
            unsupported_metrics = chosen_metrics.difference(available_metrics)
            if 'metrics' in chosen_metrics:
                print('You indicated "metrics". Here are Available Metrics\n%s\n' % ', '.join(available_metrics))
                metrics_input = input('%s' % input_string)
            elif chosen_metrics.intersection(describe):
                describe_data(df=df) if df is not None else print('Can\'t describe data without providing a DataFrame')
                # df.describe() if df is not None else print('Can\'t describe data without providing a DataFrame...')
                metrics_input = input('%s' % input_string)
            elif unsupported_metrics:
                # TODO catch value error in dict comprehension upon string input
                metrics_input = input('Metric%s "%s" not found in the DataFrame! Is your spelling correct? Have you '
                                      'used the correct underscores? Please input these metrics again. Specify '
                                      '"metrics" to view available metrics%s'
                                      % ('s' if len(unsupported_metrics) > 1 else '', ', '.join(unsupported_metrics),
                                         input_string))
            elif len(chosen_metrics) > 0:
                break
            else:
                print('No metrics were provided... If this is what you want, you can run this module without '
                      'the %s flag' % '-f/--filter' if mode == 'filter' else '-w/--weight')
                if verify_choice():
                    break
            fixed_metrics = list(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                                    repeat(' '), repeat('_'))))
            chosen_metrics = chosen_metrics.difference(unsupported_metrics).union(fixed_metrics)
            # unsupported_metrics = set(chosen_metrics).difference(available_metrics)

        print(instructions[mode])
        while True:  # not correct:  # correct = False
            print('%s' % (describe_string if df is not None else ''))
            metric_values = {}
            for metric in chosen_metrics:
                while True:
                    # Todo make ability to use boolean descriptions
                    # Todo make ability to specify direction
                    value = input('For "%s" what value should be used for %s %sing?%s%s'
                                  % (metric, level, mode, ' Designs with metrics %s than this value will be included'
                                                          % direction[filter_df.loc['direction', metric]].upper()
                                                          if mode == 'filter' else '', input_string))
                    if value in describe:
                        describe_data(df=df) if df is not None \
                            else print('Can\'t describe data without providing a DataFrame...')
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
                print('You selected:\n\t%s' % '\n\t'.join(pretty_format_table(metric_values.items())))
            else:
                # print('No metrics were provided, skipping value input')
                metric_values = None
                break

            if verify_choice():
                break
    except KeyboardInterrupt:
        exit('Selection was ended by Ctrl-C!')

    return metric_values


def rank_dataframe_by_metric_weights(df: pd.DataFrame, weights: dict[str, float] = None,
                                     function: str = Literal[rank, normalize],
                                     **kwargs) -> pd.Series:
    """From a provided DataFrame with individual design trajectories, select trajectories based on provided metric and
    weighting parameters

    Args:
        df: The designs x metrics DataFrame (single index metrics column) to select trajectories from
        weights: {'metric': value, ...}. If not provided, sorts by interface_energy
        function: The function to use for weighting. Either 'rank' or 'normalize' is possible
    Returns:
        The sorted Series of values with the best indices first (top) and the worst on the bottom
    """
    if not function:
        function = rank
    if weights:
        weights = {metric: dict(direction=filter_df.loc['direction', metric], value=value)
                   for metric, value in weights.items()}
        # This sorts the wrong direction despite the perception that it sorts correctly
        # sort_direction = dict(max=False, min=True}  # max - ascending=False, min - ascending=True
        # This sorts the correct direction, putting small and negative value (when min is better) with higher rank
        sort_direction = dict(max=True, min=False)  # max - ascending=False, min - ascending=True
        if function == rank:
            df = pd.concat({metric: df[metric].rank(ascending=sort_direction[parameters['direction']],
                                                    method=parameters['direction'], pct=True) * parameters['value']
                            for metric, parameters in weights.items()}, axis=1)
        elif function == normalize:  # get the MinMax normalization (df - df.min()) / (df.max() - df.min())
            normalized_metric_df = {}
            for metric, parameters in weights.items():
                metric_s = df[metric]
                if parameters['direction'] == _max:
                    metric_min, metric_max = metric_s.min(), metric_s.max()
                else:  # parameters['direction'] == 'min:'
                    metric_min, metric_max = metric_s.max(), metric_s.min()
                normalized_metric_df[metric] = ((metric_s - metric_min) / (metric_max - metric_min)) * parameters['value']
            df = pd.concat(normalized_metric_df, axis=1)
        else:
            raise ValueError('The value %s is not a viable choice for metric weighting "function"' % function)

        return df.sum(axis=1).sort_values(ascending=False)
    else:  # just sort by lowest energy
        return df['interface_energy'].sort_values('interface_energy', ascending=True)


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
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 1, 'Y': 1, 'B': 0, 'J': 0, 'O': 0, 'U': 0,
                  'X': 0, 'Z': 0},
     'standard': {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 1, 'K': 0, 'L': 1, 'M': 0, 'N': 0,
                  'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 1, 'W': 0, 'Y': 0, 'B': 0, 'J': 0, 'O': 0, 'U': 0,
                  'X': 0, 'Z': 0}}


def hydrophobic_collapse_index(sequence: Sequence[str | int] | np.ndarry, hydrophobicity: str = 'standard',
                               custom: dict[protein_letters_literal, int] = None, alphabet_type: alphabet_types = None,
                               lower_window: int = 3, upper_window: int = 9) -> np.ndarray:
    """Calculate hydrophobic collapse index for sequence(s) of interest and return an HCI array

    Args:
        sequence: The sequence to measure. Can be a character based sequence (or array of sequences with shape
            (sequences, residues)), an integer based sequence, or a sequence profile like array (residues, alphabet)
            where each character in the alphabet contains a typical distribution of amino acid observations
        hydrophobicity: The hydrophobicity scale to consider. Either 'standard' (FILV), 'expanded' (FMILYVW),
            or provide one with 'custom' keyword argument
        custom: A user defined dictionary of hydrophobicity key, value (float/int) pairs
        alphabet_type: The amino acid alphabet used if the sequence consists of integer characters
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

    if isinstance(sequence[0], int):  # This is an integer sequence. An alphabet is required
        print('integer')
        if alphabet_type is None:
            raise ValueError(f'Must pass an alphabet type when calculating {hydrophobic_collapse_index.__name__} using'
                             f' integer sequence values')
        else:
            alphabet = create_translation_tables(alphabet_type)

        values = [hydrophobicity_values[aa] for aa in alphabet]
        if isinstance(sequence, np.ndarray) and sequence.ndim == 2:
            print('array.shape', sequence.shape, 'values.shape', len(values))
            # The array must have shape (number_of_residues, alphabet_length)
            sequence_array = sequence * values
            # Ensure each position is a combination of the values for each amino acid
            sequence_array = sequence_array.sum(axis=-1)
        # elif isinstance(sequence, Sequence):
        #     sequence_array = [values[aa_int] for aa_int in sequence]
        else:
            sequence_array = [values[aa_int] for aa_int in sequence]
            # raise ValueError(f"sequence argument with type {type(sequence).__name__} isn't supported")
    elif isinstance(sequence[0], str):  # This is a string array # if isinstance(sequence[0], str):
        if isinstance(sequence, np.ndarray) and sequence.ndim == 2:  # (np.ndarray, list)):
            # The array must have shape (number_of_residues, alphabet_length)
            sequence_array = sequence * np.vectorize(hydrophobicity_values.__getitem__)(sequence)
            # Ensure each position is a combination of the values for each amino acid in the array
            sequence_array = sequence_array.mean(axis=-2)
        # elif isinstance(sequence, Sequence):
        #     sequence_array = [hydrophobicity_values.get(aa, 0) for aa in sequence]
        else:
            sequence_array = [hydrophobicity_values.get(aa, 0) for aa in sequence]
            # raise ValueError(f"sequence argument with type {type(sequence).__name__} isn't supported")
    else:
        raise ValueError(f'The provided sequence must comprise the canonical amino acid string characters or '
                         f'integer values corresponding to numerical amino acid conversions. '
                         f'Got type={type(sequence[0]).__name__} instead')

    window_array = window_function(sequence_array, lower=lower_window, upper=upper_window)

    return window_array.mean(axis=0)
