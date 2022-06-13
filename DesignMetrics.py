import operator
from copy import copy
from itertools import repeat
from json import loads
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd

from PathUtils import groups, reference_name, structure_background, design_profile, hbnet_design_profile
from Query.utils import input_string, validate_type, verify_choice, header_string
from SymDesignUtils import start_log, DesignError, index_intersection, handle_errors, \
    pretty_format_table, digit_translate_table

# Globals
logger = start_log(name=__name__)
metric_weight_functions = ['rank', 'normalize']
master_metrics = {'average_fragment_z_score':
                      {'description': 'The average fragment z-value used in docking/design',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  # 'buns_heavy_total':
                  #     {'description': 'Buried unsaturated H-bonding heavy atoms in the design',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
                  # 'buns_hpol_total':
                  #     {'description': 'Buried unsaturated H-bonding polarized hydrogen atoms in the design',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
                  # 'buns_total':
                  #     {'description': 'Total buried unsaturated H-bonds in the design',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
                  'buried_unsatisfied_hbond_density':
                      {'description': 'Buried Unsaturated Hbonds per Angstrom^2 of interface',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'buried_unsatisfied_hbonds':
                      {'description': 'Total buried unsatisfied H-bonds in the design',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  # 'component_1_symmetry':
                  #     {'description': 'The symmetry group of component 1',
                  #      'direction': 'min', 'function': 'equals', 'filter': True},
                  # 'component_1_name':
                  #     {'description': 'component 1 PDB_ID', 'direction': None, 'function': None, 'filter': False},
                  # 'component_1_number_of_residues':
                  #     {'description': 'The number of residues in the monomer of component 1',
                  #      'direction': 'max', 'function': 'rank', 'filter': True},
                  # 'component_1_max_radius':
                  #     {'description': 'The maximum distance that component 1 reaches away from the center of mass',
                  #      'direction': 'max', 'function': 'normalize', 'filter': True},
                  # 'component_1_n_terminal_helix':
                  #     {'description': 'Whether the n-terminus has an alpha helix',
                  #      'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  # 'component_1_c_terminal_helix':
                  #     {'description': 'Whether the c-terminus has an alpha helix',
                  #      'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  # 'component_1_n_terminal_orientation':
                  #     {'description': 'The direction the n-terminus is oriented from the symmetry group center of mass.'
                  #                     ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
                  # 'component_1_c_terminal_orientation':
                  #     {'description': 'The direction the c-terminus is oriented from the symmetry group center of mass.'
                  #                     ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
                  # 'component_2_symmetry':
                  #     {'description': 'The symmetry group of component 2',
                  #      'direction': 'min', 'function': 'equals', 'filter': True},
                  # 'component_2_name':
                  #     {'description': 'component 2 PDB_ID', 'direction': None, 'function': None, 'filter': False},
                  # 'component_2_number_of_residues':
                  #     {'description': 'The number of residues in the monomer of component 2',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
                  # 'component_2_max_radius':
                  #     {'description': 'The maximum distance that component 2 reaches away from the center of mass',
                  #      'direction': 'max', 'function': 'normalize', 'filter': True},
                  # 'component_2_n_terminal_helix':
                  #     {'description': 'Whether the n-terminus has an alpha helix',
                  #      'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  # 'component_2_c_terminal_helix':
                  #     {'description': 'Whether the c-terminus has an alpha helix',
                  #      'direction': None, 'function': None, 'filter': True},  # Todo binary?
                  # 'component_2_n_terminal_orientation':
                  #     {'description': 'The direction the n-terminus is oriented from the symmetry group center of mass.'
                  #                     ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
                  # 'component_2_c_terminal_orientation':
                  #     {'description': 'The direction the c-terminus is oriented from the symmetry group center of mass.'
                  #                     ' 1 is away, -1 is towards', 'direction': None, 'function': None, 'filter': False},
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
                  'energy_distance_from_structure_background_mean':
                      {'description': 'The distance of the design\'s per residue energy from a design with no '
                                      'constraint on amino acid selection', 'direction': 'min', 'function': 'rank',
                       'filter': True},
                  'entity_1_c_terminal_helix':  # TODO make a single metric
                      {'description': 'Whether the entity has a c-terminal helix',
                       'direction': 'max', 'function': 'boolean', 'filter': True},
                  'entity_1_c_terminal_orientation':  # TODO make a single metric
                      {'description': 'Whether the entity c-termini is closer to the assembly core or surface (1 is '
                                      'away, -1 is towards',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'entity_1_max_radius':  # TODO make a single metric
                      {'description': 'The furthest point the entity reaches from the assembly core',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_1_min_radius':  # TODO make a single metric
                      {'description': 'The closest point the entity approaches the assembly core',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'entity_1_n_terminal_helix':  # TODO make a single metric
                      {'description': 'Whether the entity has a n-terminal helix',
                       'direction': 'max', 'function': 'boolean', 'filter': True},
                  'entity_1_n_terminal_orientation':  # TODO make a single metric
                      {'description': 'Whether the entity n-termini is closer to the assembly core or surface (1 is '
                                      'away, -1 is towards)',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'entity_1_name':  # TODO make a single metric
                      {'description': 'The name of the entity', 'direction': None, 'function': None, 'filter': None},
                  'entity_1_number_of_mutations':  # TODO make a single metric
                      {'description': 'The number of mutations made',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_1_number_of_residues':  # TODO make a single metric
                      {'description': 'The number of residues', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_1_percent_mutations':  # TODO make a single metric
                      {'description': 'The percentage of the entity that has been mutated',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_1_radius':  # TODO make a single metric
                      {'description': 'The center of mass of the entity from the assembly core',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_1_symmetry':  # TODO make a single metric
                      {'description': 'The symmetry notation of the entity',
                       'direction': None, 'function': None, 'filter': None},
                  'entity_1_thermophile':  # TODO make a single metric
                      {'description': 'Whether the entity is a thermophile',
                       'direction': 'max', 'function': 'boolean', 'filter': None},
                  'entity_max_radius_average_deviation':
                      {'description': 'In a multi entity assembly, the total deviation of the max radii of each entity '
                                      'from one another', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_max_radius_ratio_1v2':  # TODO make a single metric
                      {'description': '', 'direction': None, 'function': None, 'filter': None},
                  'entity_maximum_radius':
                      {'description': 'The maximum radius any entity extends from the assembly core',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_min_radius_average_deviation':
                      {'description': 'In a multi entity assembly, the total deviation of the min radii of each entity '
                                      'from one another', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_min_radius_ratio_1v2':  # TODO make a single metric
                      {'description': '', 'direction': None, 'function': None, 'filter': None},
                  'entity_minimum_radius':
                      {'description': 'The minimum radius any entity approaches the assembly core',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'entity_number_of_residues_average_deviation':
                      {'description': 'In a multi entity assembly, the total deviation of the number of residues of '
                                      'each entity from one another',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_number_of_residues_ratio_1v2':  # TODO make a single metric
                      {'description': '', 'direction': None, 'function': None, 'filter': None},
                  'entity_radius_average_deviation':
                      {'description': 'In a multi entity assembly, the total deviation of the center of mass of each '
                                      'entity from one another',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_radius_ratio_1v2':  # TODO make a single metric
                      {'description': '', 'direction': None, 'function': None, 'filter': None},
                  'entity_residue_length_total':
                      {'description': 'The total number of residues in the design',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'entity_thermophilicity':
                      {'description': 'The extent to which the entities in the pose are thermophilic',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'errat_accuracy':
                      {'description': 'The overall Errat score of the design',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'errat_deviation':
                      {'description': 'Whether a residue window deviates significantly from typical Errat distribution',
                       'direction': 'min', 'function': 'boolean', 'filter': True},
                  'favor_residue_energy':
                      {'description': 'Total weight of sequence constraints used to favor certain amino acids in design'
                                      '. Only protocols with a favored profile have values',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  # 'fragment_z_score_total':
                  #     {'description': 'The sum of all fragments z-values',
                  #      'direction': None, 'function': None, 'filter': None},
                  'interaction_energy_complex':
                      {'description': 'The two-body (residue-pair) energy of the complexed interface. No solvation '
                                      'energies', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'global_collapse_z_sum':
                      {'description': 'The sum of collapsing sequence regions z-score. Measures the magnitude of '
                                      'additional hydrophobic collapse',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'hydrophobicity_deviation_magnitude':
                      {'description': 'The total deviation in the hydrophobic collapse, either more or less collapse '
                                      'prone', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_area_hydrophobic':
                      {'description': 'Total hydrophobic interface buried surface area',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_area_polar':
                      {'description': 'Total polar interface buried surface area',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_area_to_residue_surface_ratio':
                      {'description': 'The average ratio of interface buried surface area to the surface accessible '
                                      'residue area in the uncomplexed pose',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_area_total':
                      {'description': 'Total interface buried surface area',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_b_factor_per_residue':
                      {'description': 'The average B-factor from each atom, from each interface residue',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_bound_activation_energy':
                      {'description': 'Energy required for the unbound interface to adopt the conformation in the '
                                      'complexed state', 'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_composition_similarity':
                      {'description': 'The similarity to the expected interface composition given interface buried '
                                      'surface area. 1 is similar to natural interfaces, 0 is dissimilar',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'interface_connectivity_1':  # TODO make a single metric
                      {'description': 'How embedded is interface1 in the rest of the protein?',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  # 'interface_connectivity_2':
                  #     {'description': 'How embedded is interface2 in the rest of the protein?',
                  #      'direction': 'max', 'function': 'normalize', 'filter': True},
                  'interface_connectivity':
                      {'description': 'How embedded is the total interface in the rest of the protein?',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  # 'int_energy_density':
                  #     {'description': 'Energy in the bound complex per Angstrom^2 of interface area',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
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
                  'interface_energy_1_unbound':  # TODO make a single metric or remove
                      {'description': 'Sum of interface1 residue energy in the unbound state',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  # 'interface_energy_2_unbound':
                  #     {'description': 'Sum of interface2 residue energy in the unbound state',
                  #      'direction': 'min', 'function': 'rank', 'filter': True},
                  'interface_separation':
                      {'description': 'Median distance between all atom points on each side of the interface',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'interface_separation_core':
                      {'description': 'Median distance between all atom points on each side of the interface core '
                                      'fragment positions',
                       'direction': 'min', 'function': 'normalize', 'filter': True},
                  'interface_secondary_structure_count':
                      {'description': 'The number of unique secondary structures in the interface',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'interface_secondary_structure_fragment_count':
                      {'description': 'The number of unique fragment containing secondary structures in the interface',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  # DSSP G:310 helix, H:α helix and I:π helix, B:beta bridge, E:strand/beta bulge, T:turns,
                  #      S:high curvature (where the angle between i-2, i, and i+2 is at least 70°), and " "(space):loop
                  'interface_secondary_structure_fragment_topology':
                      {'description': 'The Stride based secondary structure names of each unique element where possible'
                                      ' values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                                      'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
                       'direction': None, 'function': None, 'filter': None},
                  'interface_secondary_structure_fragment_topology_1':  # TODO make a single metric or remove
                      {'description': 'The Stride based secondary structure names of each unique element where possible'
                                      ' values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                                      'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
                       'direction': None, 'function': None, 'filter': None},
                  'interface_secondary_structure_topology':
                      {'description': 'The Stride based secondary structure names of each unique element where possible'
                                      ' values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                                      'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
                       'direction': None, 'function': None, 'filter': None},
                  'interface_secondary_structure_topology_1':  # TODO make a single metric or remove
                      {'description': 'The Stride based secondary structure names of each unique element where possible'
                                      ' values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                                      'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
                       'direction': None, 'function': None, 'filter': None},
                  'interface_local_density':
                      {'description': 'A measure of the average number of atom neighbors for each atom in the interface'
                       , 'direction': 'max', 'function': 'rank', 'filter': True},
                  'multiple_fragment_ratio':
                      {'description': 'The extent to which fragment observations are connected in the interface. Higher'
                                      ' ratio means multiple fragment observations per residue',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score':
                      {'description': 'Sum of total fragment containing residue match scores (1 / 1 + Z-score^2) '
                                      'weighted by their ranked match score. Maximum of 2/residue',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score_center':
                      {'description': 'nanohedra_score for the central fragment residues only',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score_center_normalized':
                      {'description': 'The central Nanohedra Score normalized by number of central fragment residues',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'nanohedra_score_normalized':
                      {'description': 'The Nanohedra Score normalized by number of fragment residues',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'new_collapse_island_significance':
                      {'description': 'The significance of new collapse islands according to their inverse signficance '
                                      'towards the contact order',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'new_collapse_islands':
                      {'description': 'The number of new collapse islands found',
                       'direction': 'min', 'function': 'rank', 'filter': True},
                  'number_fragment_residues_total':
                      {'description': 'The number of residues in the interface with fragment observationsfound',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'number_fragment_residues_center':
                      {'description': 'The number of interface residues that belong to a central fragment residue',
                       'direction': 'max', 'function': 'rank', 'filter': None},
                  'number_hbonds':
                      {'description': 'The number of residues making H-bonds in the total interface. Residues may make '
                                      'more than one H-bond', 'direction': 'max', 'function': 'rank', 'filter': True},
                  'number_of_fragments':
                      {'description': 'The number of fragments found in the pose interface',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'number_of_mutations':
                      {'description': 'The number of mutations made to the pose (ie. wild-type residue to any other '
                                      'amino acid)', 'direction': 'min', 'function': 'normalize', 'filter': True},
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
                  'percent_mutations':
                      {'description': 'The percent of the design which has been mutated',
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
                  'rim':
                      {'description': 'The number of \'rim\' residues as classified by E. Levy 2010',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  # 'rmsd':
                  #     {'description': 'Root Mean Square Deviation of all CA atoms between the refined (relaxed) and '
                  #                     'designed states', 'direction': 'min', 'function': 'normalize', 'filter': True},
                  'rmsd_complex':
                      {'description': 'Root Mean Square Deviation of all CA atoms between the refined (relaxed) and '
                                      'designed states', 'direction': 'min', 'function': 'normalize', 'filter': True},
                  'rosetta_reference_energy':
                      {'description': 'Rosetta Energy Term - A metric for the unfolded energy of the protein along with'
                                      ' sequence fitting corrections',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'sequential_collapse_peaks_z_sum':
                      {'description': 'The collapse z-score for each residue scaled sequentially by the number of '
                                      'previously observed collapsable locations',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'sequential_collapse_z_sum':
                      {'description': 'The collapse z-score for each residue scaled by the proximity to sequence start',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  'shape_complementarity':
                      {'description': 'Measure of fit between two surfaces from Lawrence and Colman 1993',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'shape_complementarity_core':
                      {'description': 'Measure of fit between two surfaces from Lawrence and Colman 1993 at interface '
                                      'core fragment positions',
                       'direction': 'max', 'function': 'normalize', 'filter': True},
                  'interface_solvation_energy':  # free_energy of desolvation is positive for bound interfaces. unbound - complex
                      {'description': 'The free energy resulting from hydration of the separated interface surfaces. '
                                      'Positive values indicate poorly soluble surfaces upon dissociation',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'interface_solvation_energy_activation':  # unbound - bound
                      {'description': 'The free energy of solvation resulting from packing the bound, uncomplexed state'
                                      ' to an unbound, uncomplexed state. Positive values indicate a tendency towards '
                                      'the bound configuration',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'interface_solvation_energy_bound':
                      {'description': 'The desolvation free energy of the separated interface surfaces. Positive values'
                                      ' indicate energy is required to desolvate',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'interface_solvation_energy_complex':
                      {'description': 'The desolvation free energy of the complexed interface. Positive values'
                                      ' indicate energy is required to desolvate',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'interface_solvation_energy_unbound':
                      {'description': 'The desolvation free energy of the separated, repacked, interface surfaces. '
                                      'Positive values indicate energy is required to desolvate',
                       'direction': 'min', 'function': 'rank\n', 'filter': True},
                  'support':
                      {'description': 'The number of \'support\' residues as classified by E. Levy 2010',
                       'direction': 'max', 'function': 'rank', 'filter': True},
                  # 'symmetry':
                  #     {'description': 'The specific symmetry type used design (point (0), layer (2), lattice(3))',
                  #      'direction': None, 'function': None, 'filter': False},
                  'symmetry_group_1':  # TODO make a single metric
                      {'description': 'The specific symmetry of group 1 from of Nanohedra symmetry combination '
                                      'materials (SCM)',
                       'direction': None, 'function': None, 'filter': False},
                  # 'symmetry_group_2':
                  #     {'description': 'The specific symmetry of group 1 from of Nanohedra symmetry combination '
                  #                     'materials (SCM)',
                  #      'direction': None, 'function': None, 'filter': False},
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
# per_res_keys = ['jsd', 'des_jsd', 'int_jsd', 'frag_jsd']
fragment_metric_template = {'center_residues': set(), 'total_residues': set(),
                            'nanohedra_score': 0.0, 'nanohedra_score_center': 0.0, 'multiple_fragment_ratio': 0.0,
                            'number_fragment_residues_total': 0, 'number_fragment_residues_center': 0,
                            'number_fragments': 0, 'percent_fragment_helix': 0.0, 'percent_fragment_strand': 0.0,
                            'percent_fragment_coil': 0.0}


def read_scores(file: str | bytes, key: str = 'decoy') -> dict[str, dict[str, str]]:
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


def hbond_processing(design_scores: Dict, columns: List[str]) -> Dict[str, Set]:
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


def dirty_hbond_processing(design_scores: Dict) -> Dict[str, Set]:
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


def process_residue_info(design_residue_scores: Dict, mutations: Dict, hbonds: Dict = None) -> Dict:
    """Process energy metrics from per residue info and incorporate mutation and hydrogen bond measurements

    Args:
        design_residue_scores: {'001': {15: {'complex': -2.71, 'bound': [-1.9, 0], 'unbound': [-1.9, 0],
                                             'solv_complex': -2.71, 'solv_bound': [-1.9, 0], 'solv_unbound': [-1.9, 0],
                                             'fsp': 0., 'cst': 0.}, ...}, ...}
        mutations: {'reference': {mutation_index: {'from': 'A', 'to: 'K'}, ...},
                    '001': {mutation_index: {}, ...}, ...}
        hbonds: {'001': [34, 54, 67, 68, 106, 178], ...}
    Returns:
        {'001': {15: {'type': 'T', 'energy_delta': -2.71, 'coordinate_constraint': 0. 'residue_favored': 0., 'hbond': 0}
                 ...}, ...}
    """
    if not hbonds:
        hbonds = {}

    warn2 = False
    pose_length = len(mutations[reference_name])
    for design, residue_info in design_residue_scores.items():
        design_hbonds = hbonds.get(design, [])
        remove_residues = []
        for residue_number, data in residue_info.items():
            try:  # set residue AA type based on provided mutations
                data['type'] = mutations[design][residue_number]
            except KeyError:  # residue is not a mutation, so missing from mutations
                try:  # fill with aa from wt seq
                    data['type'] = mutations[reference_name][residue_number]
                except KeyError:  # residue is out of bounds on pose length
                    # Possibly a virtual residue or string that was processed incorrectly from the digit_translate_table
                    if not warn2:
                        logger.error('Encountered residue number "%d" which is not within the pose size "%d" and will '
                                     'be removed from processing. This is likely an error with residue processing or '
                                     'residue selection in the specified rosetta protocol. If there were warnings '
                                     'produced indicating a larger residue number than pose size, this problem was not '
                                     'addressable heuristically and something else has occurred. It is likely that this'
                                     ' residue number is not useful if you indeed have output_as_pdb_nums="true"'
                                     % (residue_number, pose_length))
                        warn2 = True
                    remove_residues.append(residue_number)
                    continue
            # set hbond data if available
            data['hbond'] = 1 if residue_number in design_hbonds else 0
            # compute the energy delta of each residue which requires summing the unbound energies
            data['bound'] = sum(data['bound'])
            data['unbound'] = sum(data['unbound'])
            data['solv_bound'] = sum(data['solv_bound'])
            data['solv_unbound'] = sum(data['solv_unbound'])
            data['energy_delta'] = data['complex'] - data['unbound']
            # data['energy_bound_activation'] = data['bound'] - data['unbound']
            data['coordinate_constraint'] = data.get('cst', 0.)
            data['residue_favored'] = data.get('fsp', 0.)
            # data.pop('energy')
            # if residue_data[residue_number]['energy'] <= hot_spot_energy:
            #     residue_data[residue_number]['hot_spot'] = 1

        # clean up any incorrect residues
        for residue in remove_residues:
            residue_info.pop(residue)

    return design_residue_scores


def mutation_conserved(residue_info: Dict, bkgnd: Dict) -> Dict:
    """Process residue mutations compared to evolutionary background. Returns 1 if residue is observed in background

    Both residue_dict and background must be same index
    Args:
        residue_info: {15: {'type': 'T', ...}, ...}
        bkgnd: {0: {'A': 0, 'R': 0, ...}, ...}
    Returns:
        conservation_dict: {15: 1, 21: 0, 25: 1, ...}
    """
    return {res: 1 if bkgnd[res][info['type']] > 0 else 0 for res, info in residue_info.items() if res in bkgnd}


def per_res_metric(sequence_metrics, key=None):
    """Find metric value average over all residues in a per residue dictionary with metric specified by key

    Args:
        sequence_metrics (dict): {16: {'S': 0.134, 'A': 0.050, ..., 'jsd': 0.732, 'int_jsd': 0.412}, ...}
    Keyword Args:
        key=None (str): Name of the metric to average
    Returns:
        (float): 0.367
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


def filter_df_for_index_by_value(df: pd.DataFrame, metrics: Dict) -> Dict:
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

        if specification == 'max':
            filtered_indices[column] = df[df[column] >= value].index.to_list()
        elif specification == 'min':
            filtered_indices[column] = df[df[column] <= value].index.to_list()

    return filtered_indices


def prioritize_design_indices(df: Union[pd.DataFrame, Union[str, bytes]], filter: Union[Dict, bool] = None,
                              weight: Union[Dict, bool] = None, protocol: Union[str, List[str]] = None, **kwargs) \
        -> pd.DataFrame:
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
            # raise DesignError('The protocol \'%s\' was not found in the set of designs...')
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

    # {column: {'direction': 'min', 'value': 0.3, 'idx_slice': ['0001', '0002', ...]}, ...}
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


def calculate_match_metrics(fragment_matches: List[Dict]) -> Dict:
    """Return the various metrics calculated by overlapping fragments at the interface of two proteins

    Args:
        fragment_matches: [{'mapped': entity1_resnum, 'match': score_term, 'paired': entity2_resnum,
                            'culster': cluster_id}, ...]
    Returns:
        {'mapped': {'center': {'residues': (set[int]), 'score': (float), 'number': (int)},
                    'total': {'residues': (set[int]), 'score': (float), 'number': (int)},
                    'match_scores': {residue number(int): (list[score (float)]), ...},
                    'index_count': {index (int): count (int), ...},
                    'multiple_ratio': (float)}
         'paired': {'center': , 'total': , 'match_scores': , 'index_count': , 'multiple_ratio': },
         'total':  {'center': {'score': , 'number': },
                    'total': {'score': , 'number': },
                    'index_count': , 'multiple_ratio': , 'observations': (int)}
         }
    """
    if not fragment_matches:
        return {}

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


def format_fragment_metrics(metrics: Dict, null: bool = False) -> Dict:
    """For a set of fragment metrics, return the formatted total fragment metrics

    Returns:
        {center_residues, total_residues,
         nanohedra_score, nanohedra_score_center, multiple_fragment_ratio, number_fragment_residues_total,
         number_fragment_residues_center, number_fragments, percent_fragment_helix, percent_fragment_strand,
         percent_fragment_coil}
    """
    if null or not metrics:
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


describe_string = '\nTo see a describtion of the data, enter \'describe\'\n'
describe = ['describe', 'desc', 'DESCRIBE', 'DESC', 'Describe', 'Desc']


def describe_data(df=None):
    """Describe the DataFrame to STDOUT"""
    print('The available metrics are located in the top row(s) of your DataFrame. Enter your selected metrics as a '
          'comma separated input. To see descriptions for only certain metrics, enter them here. '
          'Otherwise, hit \'Enter\'')
    metrics_input = input('%s' % input_string)
    chosen_metrics = set(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                            repeat(' '), repeat('_'))))
    # metrics = input('To see descriptions for only certain metrics, enter them here. Otherwise, hit \'Enter\'%s'
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
        direction = {'max': 'higher', 'min': 'lower'}
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
              'To \'%s\' designs, which metrics would you like to utilize?%s'
              % (level, mode, describe_string if df is not None else ''))

        print('The available metrics are located in the top row(s) of your DataFrame. Enter your selected metrics as a '
              'comma separated input or alternatively, you can check out the available metrics by entering \'metrics\'.'
              '\nEx: \'shape_complementarity, contact_count, etc.\'')
        metrics_input = input('%s' % input_string)
        chosen_metrics = set(map(str.lower, map(str.replace, map(str.strip, metrics_input.strip(',').split(',')),
                                                repeat(' '), repeat('_'))))
        while True:  # unsupported_metrics or 'metrics' in chosen_metrics:
            unsupported_metrics = chosen_metrics.difference(available_metrics)
            if 'metrics' in chosen_metrics:
                print('You indicated \'metrics\'. Here are Available Metrics\n%s\n' % ', '.join(available_metrics))
                metrics_input = input('%s' % input_string)
            elif chosen_metrics.intersection(describe):
                describe_data(df=df) if df is not None else print('Can\'t describe data without providing a DataFrame')
                # df.describe() if df is not None else print('Can\'t describe data without providing a DataFrame...')
                metrics_input = input('%s' % input_string)
            elif unsupported_metrics:
                # TODO catch value error in dict comprehension upon string input
                metrics_input = input('Metric%s \'%s\' not found in the DataFrame! Is your spelling correct? Have you '
                                      'used the correct underscores? Please input these metrics again. Specify '
                                      '\'metrics\' to view available metrics%s'
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
                    value = input('For \'%s\' what value should be used for %s %sing?%s%s'
                                  % (metric, level, mode, ' Designs with metrics %s than this value will be included'
                                                          % direction[filter_df.loc['direction', metric]].upper()
                                                          if mode == 'filter' else '', input_string))
                    if value in describe:
                        describe_data(df=df) if df is not None \
                            else print('Can\'t describe data without providing a DataFrame...')
                    elif validate_type(value, dtype=float):
                        metric_values[metric] = float(value)
                        break

            # metric_values = {metric: float(input('For \'%s\' what value should be used for %s %sing?%s%s'
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


def rank_dataframe_by_metric_weights(df: pd.DataFrame, weights: Dict[str, float] = None, function: str = 'rank',
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
        function = 'rank'
    if weights:
        weights = {metric: {'direction': filter_df.loc['direction', metric], 'value': value}
                   for metric, value in weights.items()}
        # This sorts the wrong direction despite the perception that it sorts correctly
        # sort_direction = {'max': False, 'min': True}  # max - ascending=False, min - ascending=True
        # This sorts the correct direction, putting small and negative value (when min is better) with higher rank
        sort_direction = {'max': True, 'min': False}  # max - ascending=False, min - ascending=True
        if function == 'rank':
            df = pd.concat({metric: df[metric].rank(ascending=sort_direction[parameters['direction']],
                                                    method=parameters['direction'], pct=True) * parameters['value']
                            for metric, parameters in weights.items()}, axis=1)
        elif function == 'normalize':  # get the MinMax normalization (df - df.min()) / (df.max() - df.min())
            normalized_metric_df = {}
            for metric, parameters in weights.items():
                metric_s = df[metric]
                if parameters['direction'] == 'max':
                    metric_min, metric_max = metric_s.min(), metric_s.max()
                else:  # parameters['direction'] == 'min:'
                    metric_min, metric_max = metric_s.max(), metric_s.min()
                normalized_metric_df[metric] = ((metric_s - metric_min) / (metric_max - metric_min)) * parameters['value']
            df = pd.concat(normalized_metric_df, axis=1)
        else:
            raise ValueError('The value %s is not a viable choice for metric weighting \'function\'' % function)

        return df.sum(axis=1).sort_values(ascending=False)
    else:  # just sort by lowest energy
        return df['interface_energy'].sort_values('interface_energy', ascending=True)
