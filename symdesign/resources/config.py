from __future__ import annotations

import csv
from typing import Literal, get_args

from symdesign.utils import path as putils

_min, _max = 'min', 'max'
rank, normalize, boolean = 'rank', 'normalize', 'boolean'
weight_functions_literal = Literal['rank', 'normalize']
metric_weight_functions: tuple[weight_functions_literal, ...] = get_args(weight_functions_literal)

metrics = {
    'area_hydrophobic_complex':
        dict(description='Total hydrophobic solvent accessible surface area in the complexed state',
             direction=_min, function=rank, filter=True),
    'area_hydrophobic_unbound':
        dict(description='Total hydrophobic solvent accessible surface area in the unbound state',
             direction=_min, function=rank, filter=True),
    'area_polar_complex': dict(description='Total polar solvent accessible surface area in the complexed state',
                               direction=_min, function=rank, filter=True),
    'area_polar_unbound': dict(description='Total polar solvent accessible surface area in the unbound state',
                               direction=_min, function=rank, filter=True),
    'area_total_complex': dict(description='Total solvent accessible surface area in the complexed state',
                               direction=_min, function=rank, filter=True),
    'area_total_unbound': dict(description='Total solvent accessible surface area in the unbound state',
                               direction=_min, function=rank, filter=True),
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
    'buns_unbound':
        dict(description='Total buried unsaturated H-bonds in the design',
             direction=_min, function=rank, filter=True),
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
    'collapse_deviation_magnitude':
        dict(description='The total deviation in the hydrophobic collapse. Either more or less collapse prone',
             direction=_min, function=rank, filter=True),
    'collapse_variance':
        dict(description='The average/expected deviation of the hydrophobic collapse from a reference collapse',
             direction=_min, function=rank, filter=True),
    'collapse_increase_significance_by_contact_order_z':
        dict(description='Summation of positions with increased collapse from reference scaled by the inverse contact '
                         'order z-score. Where positive is more isolated collapse, and negative indicates collapse '
                         'occurs in higher contact order sites. More significant collapse is more positive',
             direction=_min, function=rank, filter=True),
    'collapse_increased_z':
        dict(description='The sum of all sequence regions z-scores experiencing increased collapse. Measures the '
                         'normalized magnitude of additional hydrophobic collapse',
             direction=_min, function=rank, filter=True),
    'collapse_increased_z_mean':
        dict(description='Mean of the collapse_increased_z per-position experiencing increased collapse',
             direction=_min, function=rank, filter=True),
    'collapse_new_position_significance':
        dict(description='The magnitude of the collapse_significance_by_contact_order_z (abs(deviation)) for identified'
                         ' new collapse positions',
             direction=_min, function=rank, filter=True),
    'collapse_new_positions':
        dict(description='The number of new collapse positions found',
             direction=_min, function=rank, filter=True),
    'collapse_sequential_peaks_z':
        dict(description='Summation of the collapse z-score for each residue scaled sequentially by the number of '
                         'previously observed collapsable locations',
             direction=_min, function=rank, filter=True),
    'collapse_sequential_peaks_z_mean':
        dict(description='Mean of the collapse_sequential_peaks_z per-position experiencing increased collapse',
             direction=_min, function=rank, filter=True),
    'collapse_sequential_z':
        dict(description='Summation of the collapse z-score for each residue scaled by the proximity to sequence start',
             direction=_min, function=rank, filter=True),
    'collapse_sequential_z_mean':
        dict(description='Mean of the collapse_sequential_z per-position experiencing increased collapse',
             direction=_min, function=rank, filter=True),
    'collapse_significance_by_contact_order_z':
        dict(description='Summed significance. Takes the product of collapse z-score at collapsing positions and '
                         'contact order per residue. Resulting values are positive when collapse occurs in areas with '
                         'low contact order, and negative when collapse occurs in high contact order positions. A '
                         'protein fold with high contact order may tolerate collapse differently than low contact order'
                         ", where the segment would rely on it's collapse to fold",
             direction=_min, function=rank, filter=True),
    'collapse_significance_by_contact_order_z_mean':
        dict(description='Mean of the collapse_significance_by_contact_order_z per-position experiencing collapse',
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
        dict(description="The distance of the design's per residue energy from a design with no constraint on "
                         'amino acid selection',
             direction=_min, function=rank, filter=True),
    # 'entity_1_c_terminal_helix':
    #     dict(description='Whether the entity has a c-terminal helix',
    #          direction=_max, function=boolean, filter=True),
    # 'entity_1_c_terminal_orientation':
    #     dict(description='Whether the entity c-termini is closer to the assembly core or surface (1 is away, -1 is '
    #                      'towards',
    #          direction=_max, function=rank, filter=True),
    # 'entity_1_max_radius':
    #     dict(description='The furthest point the entity reaches from the assembly core',
    #          direction=_min, function=rank, filter=True),
    # 'entity_1_min_radius':
    #     dict(description='The closest point the entity approaches the assembly core',
    #          direction=_max, function=rank, filter=True),
    # 'entity_1_n_terminal_helix':
    #     dict(description='Whether the entity has a n-terminal helix',
    #          direction=_max, function=boolean, filter=True),
    # 'entity_1_n_terminal_orientation':
    #     dict(description='Whether the entity n-termini is closer to the assembly core or surface (1 is away, -1 is '
    #                      'towards)',
    #          direction=_max, function=rank, filter=True),
    # 'entity_1_name':
    #     dict(description='The name of the entity',
    #          direction=None, function=None, filter=None),
    # 'entity_1_number_of_mutations':
    #     dict(description='The number of mutations made',
    #          direction=_min, function=rank, filter=True),
    # 'entity_1_number_of_residues':
    #     dict(description='The number of residues',
    #          direction=_min, function=rank, filter=True),
    # 'entity_1_percent_mutations':
    #     dict(description='The percentage of the entity that has been mutated',
    #          direction=_min, function=rank, filter=True),
    # 'entity_1_radius':
    #     dict(description='The center of mass of the entity from the assembly core',
    #          direction=_min, function=rank, filter=True),
    # 'entity_1_symmetry':
    #     dict(description='The symmetry notation of the entity',
    #          direction=None, function=None, filter=None),
    # 'entity_1_thermophile':
    #     dict(description='Whether the entity is a thermophile',
    #          direction=_max, function=boolean, filter=None),
    # 'entity_max_radius_ratio_1v2':
    #     dict(description='The ratio of the maximum radius from a reference of component 1 versus 2',
    #          direction=None, function=None, filter=None),
    # 'entity_min_radius_ratio_1v2':
    #     dict(description='The ratio of the minimum radius from a reference of component 1 versus 2',
    #          direction=None, function=None, filter=None),
    # 'entity_number_of_residues_ratio_1v2':
    #     dict(description='', direction=None, function=None, filter=None),
    # 'entity_radius_ratio_1v2':
    #     dict(description='', direction=None, function=None, filter=None),
    # START entity#_ metrics. These are used with numbers during actual metric acquisition
    'entity_c_terminal_helix':
        dict(description='Whether the entity has a c-terminal helix',
             direction=_max, function=boolean, filter=True),
    'entity_c_terminal_orientation':
        dict(description='Whether the entity c-termini is closer to the assembly core or surface (1 is away, -1 is '
                         'towards',
             direction=_max, function=rank, filter=True),
    'entity_max_radius':
        dict(description='The furthest point the entity reaches from the assembly core',
             direction=_min, function=rank, filter=True),
    'entity_min_radius':
        dict(description='The closest point the entity approaches the assembly core',
             direction=_max, function=rank, filter=True),
    'entity_n_terminal_helix':
        dict(description='Whether the entity has a n-terminal helix',
             direction=_max, function=boolean, filter=True),
    'entity_n_terminal_orientation':
        dict(description='Whether the entity n-termini is closer to the assembly core or surface (1 is away, -1 is '
                         'towards)',
             direction=_max, function=rank, filter=True),
    'entity_name':
        dict(description='The name of the entity',
             direction=None, function=None, filter=None),
    'entity_number_of_mutations':
        dict(description='The number of mutations made',
             direction=_min, function=rank, filter=True),
    'entity_number_of_residues':
        dict(description='The number of residues',
             direction=_min, function=rank, filter=True),
    'entity_percent_mutations':
        dict(description='The percentage of the entity that has been mutated',
             direction=_min, function=rank, filter=True),
    'entity_radius':
        dict(description='The center of mass of the entity from the assembly core',
             direction=_min, function=rank, filter=True),
    'entity_symmetry_group':
        dict(description='The symmetry notation of the entity',
             direction=None, function=None, filter=True),
    'entity_thermophile':
        dict(description='Whether the entity is a thermophile',
             direction=_max, function=boolean, filter=None),
    'entity_max_radius_ratio_v':
        dict(description='The ratio of the maximum radius from a reference of component 1 versus 2',
             direction=None, function=None, filter=None),
    'entity_min_radius_ratio_v':
        dict(description='The ratio of the minimum radius from a reference of component 1 versus 2',
             direction=None, function=None, filter=None),
    'entity_number_of_residues_ratio_v':
        dict(description='', direction=None, function=None, filter=None),
    'entity_interface_connectivity':
        dict(description='How embedded is the entity interface in the rest of the protein?',
             direction=_max, function=normalize, filter=True),
    'entity_interface_secondary_structure_fragment_topology':
        dict(description='The Stride based secondary structure names of each unique element where possible '
                         'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                         'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
             direction=None, function=None, filter=None),
    'entity_interface_secondary_structure_topology':
        dict(description='The Stride based secondary structure names of each unique element where possible '
                         'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                         'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
             direction=None, function=None, filter=None),
    'entity_radius_ratio_v':
        dict(description='', direction=None, function=None, filter=None),
    # END entity#_ metrics
    'entity_max_radius_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the max radii of each entity '
                         'from one another', direction=_min, function=rank, filter=True),
    'entity_min_radius_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the min radii of each entity from'
                         ' one another',
             direction=_min, function=rank, filter=True),
    'entity_number_of_residues_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the number of residues of each '
                         'entity from one another',
             direction=_min, function=rank, filter=True),
    'entity_radius_average_deviation':
        dict(description='In a multi entity assembly, the total deviation of the center of mass of each entity'
                         ' from one another',
             direction=_min, function=rank, filter=True),
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
    # 'interface_connectivity1':
    #     dict(description='How embedded is interface1 in the rest of the protein?',
    #          direction=_max, function=normalize, filter=True),
    # 'interface_connectivity2':
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
    # 'interface_energy_1_unbound':
    #     dict(description='Sum of interface1 residue energy in the unbound state',
    #          direction=_min, function=rank, filter=True),
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
    'interface_secondary_structure_topology':
        dict(description='The Stride based secondary structure names of each unique element where possible '
                         'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                         'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
             direction=None, function=None, filter=None),
    'interface_local_density':
        dict(description='A measure of the average number of atom neighbors for each atom in the interface',
             direction=_max, function=rank, filter=True),
    'maximum_radius':
        dict(description='The maximum radius any entity extends from the assembly core',
             direction=_min, function=rank, filter=True),
    'minimum_radius':
        dict(description='The minimum radius any entity approaches the assembly core',
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
    'number_fragment_residues_total':
        dict(description='The number of residues in the interface with fragment observationsfound',
             direction=_max, function=rank, filter=True),
    'number_fragment_residues_center':
        dict(description='The number of interface residues that belong to a central fragment residue',
             direction=_max, function=rank, filter=None),
    'number_design_residues':
        dict(description='The number of residues selected for sequence design',
             direction=max, function=normalize, filter=True),
    'number_interface_residues':
        dict(description='The total number of interface residues found in the pose (default is residue CB within 8A)',
             direction=_max, function=rank, filter=True),
    'number_of_hbonds':
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
        dict(description='Percent of designed residues found in the combined profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_design_mean_probability':
        dict(description='The mean probability of the designed residues in the design profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_evolution':
        dict(description='Percent of designed residues found in the evolutionary profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_evolution_mean_probability':
        dict(description='The mean probability of the designed residues in the evolutionary profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_fragment':
        dict(description='Percent of designed residues found in the fragment profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_fragment_mean_probability':
        dict(description='The mean probability of the designed residues in the fragment profile. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_interface':
        dict(description='Percent of designed residues found in the interface fragment library. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'observed_interface_mean_probability':
        dict(description='The mean probability of the interface residues in the interface fragment library. 1 is 100%',
             direction=_max, function=rank, filter=True),
    'percent_core':
        dict(description='The percentage of residues which are "core" according to Levy, E. 2010',
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
    'percent_residues_fragment_interface_center':
        dict(description='The percentage of residues which are central fragment observations',
             direction=_max, function=normalize, filter=True),
    'percent_residues_fragment_interface_total':
        dict(description='The percentage of residues which are represented by fragment observations',
             direction=_max, function=normalize, filter=True),
    'percent_rim':
        dict(description='The percentage of residues which are "rim" according to Levy, E. 2010',
             direction=_min, function=normalize, filter=True),
    'percent_support':
        dict(description='The percentage of residues which are "support" according to Levy, E. 2010',
             direction=_max, function=normalize, filter=True),
    'pose_length':
        dict(description='The total number of residues in the design',
             direction=_min, function=rank, filter=True),
    'pose_thermophilicity':
        dict(description='The extent to which the entities in the pose are thermophilic',
             direction=_max, function=rank, filter=True),
    'proteinmpnn_v_design_probability_cross_entropy_loss':
        dict(description='The total loss between the ProteinMPNN probabilities and the design profile probabilities',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_v_design_probability_cross_entropy_score':
        dict(description='The per-designed residue cross entropy loss between the ProteinMPNN probabilities and the '
                         'design profile probabilities',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_v_evolution_probability_cross_entropy_loss':
        dict(description='The total loss between the ProteinMPNN probabilities and the evolution profile probabilities',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_v_evolution_probability_cross_entropy_score':
        dict(description='The per-designed residue cross entropy loss between the ProteinMPNN probabilities and the '
                         'evolution profile probabilities',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_v_fragment_probability_cross_entropy_loss':
        dict(description='The total loss between the ProteinMPNN probabilities and the fragment profile probabilities',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_v_fragment_probability_cross_entropy_score':
        dict(description='The per-designed residue cross entropy loss between the ProteinMPNN probabilities and the '
                         'fragment profile probabilities',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_score_delta': dict(description='The per-residue average complex-unbound ProteinMPNN score',
                                    direction=max, function=normalize, filter=True),
    'proteinmpnn_score_complex': dict(description='The per-residue average complexed ProteinMPNN score',
                                      direction=max, function=normalize, filter=True),
    'proteinmpnn_score_unbound': dict(description='The per-residue average unbound ProteinMPNN score',
                                      direction=max, function=normalize, filter=True),
    'proteinmpnn_score_delta_per_designed_residue':
        dict(description='The difference between the average complexed and unbound proteinmpnn score for designed '
                         'residues',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_score_complex_per_designed_residue':
        dict(description='The average complexed proteinmpnn score for designed residues',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_score_unbound_per_designed_residue':
        dict(description='The average unbound proteinmpnn score for designed residues',
             direction=max, function=normalize, filter=True),
    'proteinmpnn_loss_complex':
        dict(description='The magnitude of information missing from the sequence in the complexed state',
             direction=_min, function=normalize, filter=True),
    'proteinmpnn_loss_unbound':
        dict(description='The magnitude of information missing from the sequence in the unbound state',
             direction=_min, function=normalize, filter=True),
    'proteinmpnn_loss_design':
        dict(description='The magnitude of information missing from the sequence compared to the design profile',
             direction=_min, function=normalize, filter=True),
    'proteinmpnn_loss_design_per_residue':
        dict(description='The magnitude of information missing from the sequence compared to the design profile on a '
                         'per-residue basis',
             direction=_min, function=normalize, filter=True),
    'proteinmpnn_loss_evolution':
        dict(description='The magnitude of information missing from the sequence compared to the evolution profile',
             direction=_min, function=normalize, filter=True),
    'proteinmpnn_loss_evolution_per_residue':
        dict(description='The magnitude of information missing from the sequence compared to the evolution profile on a'
                         ' per-residue basis',
             direction=_min, function=normalize, filter=True),
    'proteinmpnn_loss_fragment':
        dict(description='The magnitude of information missing from the sequence compared to the fragment profile',
             direction=_min, function=normalize, filter=True),
    'proteinmpnn_loss_fragment_per_residue':
        dict(description='The magnitude of information missing from the sequence compared to the fragment profile on a '
                         'per-residue basis. Only counts residues with fragment information',
             direction=_min, function=normalize, filter=True),
    putils.protocol:
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
    'sequence':
        dict(description='The amino acid sequence of the design',
             direction=None, function=None, filter=False),
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

with open(putils.affinity_tags, 'r') as f:
    expression_tags = {'_'.join(map(str.lower, row[0].split())): row[1] for row in csv.reader(f)}
