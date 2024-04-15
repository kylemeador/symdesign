from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, get_args

import attrs
from sqlalchemy import Column

from .sql import DesignEntityMetrics, DesignMetrics, EntityMetrics, PoseMetrics, PoseResidueMetrics, ResidueMetrics
from symdesign.resources.sql import DesignProtocol, EntityTransform, JobProtocol, PoseMetadata, ProteinMetadata

program_name = Path(__file__).parent.parent.name
direction_literal = Literal['min', 'max']
weight_functions_literal = Literal['normalize', 'rank']
functions_literal = Literal[weight_functions_literal, 'boolean', 'equals']
metric_weight_functions: tuple[weight_functions_literal, ...] = get_args(weight_functions_literal)
ANGSTROM = r'$\AA$'  # Å
DELTA = '\u0394'  # Δ


@attrs.frozen(auto_attribs=True)
class Metric:
    name: str
    orm: Column
    description: str
    direction: direction_literal | None
    filter: bool | None
    function: functions_literal | None = None

    def get_filters(self) -> dict[str, Any]:
        """Get the attributes used during filtering"""
        return {'direction': self.direction, 'filter': self.filter, 'function': self.function}


# metrics = [
rotation_x = Metric(
    name='rotation_x', orm=EntityTransform.rotation_x,
    description='The x component of the internal rotation vector (in degrees)',
    direction=None, function=None, filter=True
)
rotation_y = Metric(
    name='rotation_y', orm=EntityTransform.rotation_y,
    description='The y component of the internal rotation vector (in degrees)',
    direction=None, function=None, filter=True
)
rotation_z = Metric(
    name='rotation_z', orm=EntityTransform.rotation_z,
    description='The z component of the internal rotation vector (in degrees)',
    direction=None, function=None, filter=True
)
setting_matrix = Metric(
    name='setting_matrix', orm=EntityTransform.setting_matrix,
    description='The setting_matrix transformation parameter',
    direction=None, function=None, filter=True
)
internal_translation_x = Metric(
    name='internal_translation_x', orm=EntityTransform.internal_translation_x,
    description='The internal_translation transformation parameter',
    direction=None, function=None, filter=True
)
internal_translation_y = Metric(
    name='internal_translation_y', orm=EntityTransform.internal_translation_y,
    description='The internal_translation transformation parameter',
    direction=None, function=None, filter=True
)
internal_translation_z = Metric(
    name='internal_translation_z', orm=EntityTransform.internal_translation_z,
    description='The internal_translation transformation parameter',
    direction=None, function=None, filter=True
)
external_translation_x = Metric(
    name='external_translation_x', orm=EntityTransform.external_translation_x,
    description='The external_translation_x transformation parameter',
    direction=None, function=None, filter=True
)
external_translation_y = Metric(
    name='external_translation_y', orm=EntityTransform.external_translation_y,
    description='The external_translation_y transformation parameter',
    direction=None, function=None, filter=True
)
external_translation_z = Metric(
    name='external_translation_z', orm=EntityTransform.external_translation_z,
    description='The external_translation_z transformation parameter',
    direction=None, function=None, filter=True
)
alphafold_model = Metric(
    name='alphafold_model', orm=DesignProtocol.alphafold_model,
    description='The identifier of the AlphaFold model/parameters that was used for predict-structures',
    direction=None, function='equals', filter=True
)
area_hydrophobic_complex = Metric(
    name='area_hydrophobic_complex', orm=DesignMetrics.area_hydrophobic_complex,
    description='Total hydrophobic solvent accessible surface area in the complexed state',
    direction='min', function='rank', filter=True
)
area_hydrophobic_unbound = Metric(
    name='area_hydrophobic_unbound', orm=DesignMetrics.area_hydrophobic_unbound,
    description='Total hydrophobic solvent accessible surface area in the unbound state',
    direction='min', function='rank', filter=True
)
area_polar_complex = Metric(
    name='area_polar_complex', orm=DesignMetrics.area_polar_complex,
    description='Total polar solvent accessible surface area in the complexed state',
    direction='min', function='rank', filter=True
)
area_polar_unbound = Metric(
    name='area_polar_unbound', orm=DesignMetrics.area_polar_unbound,
    description='Total polar solvent accessible surface area in the unbound state',
    direction='min', function='rank', filter=True
)
area_total_complex = Metric(
    name='area_total_complex', orm=DesignMetrics.area_total_complex,
    description='Total solvent accessible surface area in the complexed state',
    direction='min', function='rank', filter=True
)
area_total_unbound = Metric(
    name='area_total_unbound', orm=DesignMetrics.area_total_unbound,
    description='Total solvent accessible surface area in the unbound state',
    direction='min', function='rank', filter=True
)
average_fragment_z_score = Metric(
    name='average_fragment_z_score', orm=None,
    description='The average fragment z-value used in docking/design',
    direction='min', function='normalize', filter=True
)
# buns_heavy_total = Metric(
#     name='buns_heavy_total', orm=,
#     description='Buried unsaturated H-bonding heavy atoms in the design',
#          direction='min', function='rank', filter=True
# )
# buns_hpol_total = Metric(
#     name='buns_hpol_total', orm=,
#     description='Buried unsaturated H-bonding polarized hydrogen atoms in the design',
#          direction='min', function='rank', filter=True
# )
# buns_total = Metric(
#     name='buns_total', orm=,
#     description='Total buried unsaturated H-bonds in the design',
#          direction='min', function='rank', filter=True
# )
buried_unsatisfied_hbond_density = Metric(
    name='buried_unsatisfied_hbond_density', orm=DesignMetrics.buried_unsatisfied_hbond_density,
    description=f'Buried unsatisfied H-bonds per {ANGSTROM}\N{SUPERSCRIPT TWO} of interface',
    direction='min', function='normalize', filter=True
)
buried_unsatisfied_hbonds = Metric(
    name='buried_unsatisfied_hbonds', orm=DesignMetrics.buried_unsatisfied_hbonds,
    description='Total buried unsatisfied H-bonds in the design interface',
    direction='min', function='rank', filter=True
)
buried_unsatisfied_hbonds_complex = Metric(
    name='buried_unsatisfied_hbonds_complex', orm=DesignMetrics.buried_unsatisfied_hbonds_complex,
    description='Buried unsatisfied H-bonds in the complexed interface state',
    direction='min', function='rank', filter=True
)
buried_unsatisfied_hbonds_unbound = Metric(
    name='buried_unsatisfied_hbonds_unbound', orm=None,
    # DesignMetrics.buried_unsatisfied_hbonds_unbound
    description='Buried unsatisfied H-bonds in the unbound interface state',
    direction='min', function='rank', filter=True
)
ca_only = Metric(
    name='ca_only', orm=JobProtocol.ca_only,
    description='True if a Ca representation was used for designed',
    direction=None, function='boolean', filter=True
)
commit = Metric(
    name='commit', orm=JobProtocol.commit,
    description=f'The git commit of the {program_name} source code',
    direction=None, function='equals', filter=True
)
contact_count = Metric(
    name='contact_count', orm=DesignMetrics.contact_count,
    description='Number of carbon-carbon contacts across interface',
    direction='max', function='rank', filter=True
)
contact_order = Metric(
    name='contact_order', orm=DesignMetrics.contact_order,
    description='The distance of contacts to other residues in the structure',
    direction='max', function='rank', filter=True
)
contiguous_ghosts = Metric(
    name='contiguous_ghosts', orm=JobProtocol.contiguous_ghosts,
    description='Whether ghost fragments were filtered for contiguous occurrences during docking',
    direction=None, function='boolean', filter=True
)
collapse_deviation_magnitude = Metric(
    name='collapse_deviation_magnitude',
    orm=ResidueMetrics.collapse_deviation_magnitude,
    description='The total deviation in the hydrophobic collapse. Either more or less collapse prone. '
                       'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_increase_significance_by_contact_order_z = Metric(
    name='collapse_increase_significance_by_contact_order_z', orm=DesignMetrics.collapse_increase_significance_by_contact_order_z,
    description='Summation of positions with increased collapse from reference scaled by the inverse contact '
                'order z-score. Where positive is more isolated collapse, and negative indicates collapse '
                'occurs in higher contact order sites. More significant collapse is more positive. '
                'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_increased_z = Metric(
    name='collapse_increased_z', orm=DesignMetrics.collapse_increased_z,
    description='The sum of all sequence regions z-scores experiencing increased collapse. Measures the magnitude of '
                'additional hydrophobic collapse. Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_increased_z_mean = Metric(
    name='collapse_increased_z_mean', orm=DesignMetrics.collapse_increased_z_mean,
    description='Mean of the collapse_increased_z per-position experiencing increased collapse, i.e. normalized per '
                'the number of positions experiencing additional collapse. Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_new_position_significance = Metric(
    name='collapse_new_position_significance',
    orm=DesignMetrics.collapse_new_position_significance,
    description='The magnitude of the collapse_significance_by_contact_order_z (abs(deviation)) for identified new '
                'collapse positions. Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_new_positions = Metric(
    name='collapse_new_positions', orm=DesignMetrics.collapse_new_positions,
    description='The number of new collapse positions found. Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_sequential_peaks_z = Metric(
    name='collapse_sequential_peaks_z',
    orm=DesignMetrics.collapse_sequential_peaks_z,
    description='Summation of the collapse z-score for each residue scaled sequentially by the number of previously '
                'observed collapsable locations. Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_sequential_peaks_z_mean = Metric(
    name='collapse_sequential_peaks_z_mean',
    orm=DesignMetrics.collapse_sequential_peaks_z_mean,
    description='Mean of the collapse_sequential_peaks_z per-position experiencing increased collapse. '
                'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_sequential_z = Metric(
    name='collapse_sequential_z', orm=DesignMetrics.collapse_sequential_z,
    description='Summation of the collapse z-score for each residue scaled by the proximity to sequence start.'
                ' Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_sequential_z_mean = Metric(
    name='collapse_sequential_z_mean', orm=DesignMetrics.collapse_sequential_z_mean,
    description='Mean of the collapse_sequential_z per-position experiencing increased collapse. '
                'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_significance_by_contact_order_z = Metric(
    name='collapse_significance_by_contact_order_z',
    orm=DesignMetrics.collapse_significance_by_contact_order_z,
    description='Summed significance. Takes the product of collapse z-score at collapsing positions and '
                'contact order per residue. Resulting values are positive when collapse occurs in areas with '
                'low contact order, and negative when collapse occurs in high contact order positions. A '
                'protein fold with high contact order may tolerate collapse differently than low contact order'
                ", where the segment would rely on it's collapse to fold. "
                'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_significance_by_contact_order_z_mean = Metric(
    name='collapse_significance_by_contact_order_z_mean', orm=DesignMetrics.collapse_significance_by_contact_order_z_mean,
    description='Mean of the collapse_significance_by_contact_order_z per-position experiencing collapse. '
                'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_variance = Metric(
    name='collapse_variance', orm=DesignMetrics.collapse_variance,
    description='The average/expected deviation of the hydrophobic collapse from a reference collapse. '
                'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)
collapse_violation = Metric(
    name='collapse_violation', orm=DesignMetrics.collapse_violation,
    description='Whether there are collapse_new_positions and the collapse profile is altered. '
                'Derived from PMID:28507157',
    direction='min', function='rank', filter=True
)  # Boolean
core = Metric(
    name='core', orm=DesignMetrics.core,
    description="The number of 'core' residues as classified by E. Levy 2010",
    direction='max', function='rank', filter=True
)
coordinate_constraint = Metric(
    name='coordinate_constraint', orm=DesignMetrics.coordinate_constraint,
    description='Total weight of coordinate constraints to keep design from moving in cartesian space',
    direction='min', function='normalize', filter=True
)
dock_collapse_deviation_magnitude = Metric(
    name='dock_collapse_deviation_magnitude', orm=PoseResidueMetrics.dock_collapse_deviation_magnitude,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the total '
                'deviation in the hydrophobic collapse. Either more or less collapse prone',
    direction='min', function='rank', filter=True
)
dock_collapse_increase_significance_by_contact_order_z = Metric(
    name='dock_collapse_increase_significance_by_contact_order_z', orm=PoseMetrics.dock_collapse_increase_significance_by_contact_order_z,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the summation '
                'of positions with increased collapse from reference scaled by the inverse contact order z-score. '
                'Where positive is more isolated collapse, and negative indicates collapse occurs in higher contact '
                'order sites. More significant collapse is more positive',
    direction='min', function='rank', filter=True
)
dock_collapse_increased_z = Metric(
    name='dock_collapse_increased_z', orm=PoseMetrics.dock_collapse_increased_z,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the sum of '
                'all sequence regions z-scores experiencing increased collapse. Measures the normalized magnitude of '
                'additional hydrophobic collapse',
    direction='min', function='rank', filter=True
)
dock_collapse_increased_z_mean = Metric(
    name='dock_collapse_increased_z_mean', orm=PoseMetrics.dock_collapse_increased_z_mean,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the'
                ' mean of the collapse_increased_z per-position experiencing increased collapse',
    direction='min', function='rank', filter=True
)
dock_collapse_new_position_significance = Metric(
    name='dock_collapse_new_position_significance', orm=PoseMetrics.dock_collapse_new_position_significance,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the'
                ' magnitude of the collapse_significance_by_contact_order_z (abs(deviation)) for identified'
                ' new collapse positions',
    direction='min', function='rank', filter=True
)
dock_collapse_new_positions = Metric(
    name='dock_collapse_new_positions', orm=PoseMetrics.dock_collapse_new_positions,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the'
                       ' number of new collapse positions found',
    direction='min', function='rank', filter=True
)
dock_collapse_sequential_peaks_z = Metric(
    name='dock_collapse_sequential_peaks_z', orm=PoseMetrics.dock_collapse_sequential_peaks_z,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the'
                       ' summation of the collapse z-score for each residue scaled sequentially by the number of '
                       'previously observed collapsable locations',
    direction='min', function='rank', filter=True
)
dock_collapse_sequential_peaks_z_mean = Metric(
    name='dock_collapse_sequential_peaks_z_mean', orm=PoseMetrics.dock_collapse_sequential_peaks_z_mean,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the'
                       ' mean of the collapse_sequential_peaks_z per-position experiencing increased collapse',
    direction='min', function='rank', filter=True
)
dock_collapse_sequential_z = Metric(
    name='dock_collapse_sequential_z', orm=PoseMetrics.dock_collapse_sequential_z,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the'
                       ' summation of the collapse z-score for each residue scaled by the proximity to sequence start',
    direction='min', function='rank', filter=True
)
dock_collapse_sequential_z_mean = Metric(
    name='dock_collapse_sequential_z_mean', orm=PoseMetrics.dock_collapse_sequential_z_mean,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the'
                       ' mean of the collapse_sequential_z per-position experiencing increased collapse',
    direction='min', function='rank', filter=True
)
dock_collapse_significance_by_contact_order_z = Metric(
    name='dock_collapse_significance_by_contact_order_z', orm=PoseMetrics.dock_collapse_significance_by_contact_order_z,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the '
                       'summed significance. Takes the product of collapse z-score at collapsing positions and '
                       'contact order per residue. Resulting values are positive when collapse occurs in areas with '
                       'low contact order, and negative when collapse occurs in high contact order positions. A '
                       'protein fold with high contact order may tolerate collapse differently than low contact order'
                       ", where the segment would rely on it's collapse to fold",
    direction='min', function='rank', filter=True
)
dock_collapse_significance_by_contact_order_z_mean = Metric(
    name='dock_collapse_significance_by_contact_order_z_mean', orm=PoseMetrics.dock_collapse_significance_by_contact_order_z_mean,
    description='For the docked pose scored by ProteinMPNN, uses the sequence probabilities to calculate the '
                       'mean of the collapse_significance_by_contact_order_z per-position experiencing collapse',
    direction='min', function='rank', filter=True
)
dock_collapse_variance = Metric(
    name='dock_collapse_variance', orm=PoseMetrics.dock_collapse_variance,
    description='For the ProteinMPNN structure profile, calculate the'
                       ' average/expected deviation of the hydrophobic collapse from a reference collapse',
    direction='min', function='rank', filter=True
)
dock_collapse_violation = Metric(
    name='dock_collapse_violation', orm=PoseMetrics.dock_collapse_violation,
    description='Whether there are dock_collapse_new_positions and the collapse profile is altered',
    direction='min', function='rank', filter=True
)  # Boolean
dock_hydrophobicity = Metric(
    name='dock_hydrophobicity', orm=PoseMetrics.dock_hydrophobicity,
    description='The sum of all hydrophobicity values calculated from the docking collapse profile',
    direction='min', function='rank', filter=True
)
divergence_design_per_residue = Metric(
    name='divergence_design_per_residue', orm=None,
    description='The Jensen-Shannon divergence of interface residues from the position specific design profile values. '
                'Includes fragment & evolution if both are True, otherwise only includes those specified for use in '
                'design.',
    direction='min', function='rank', filter=True
)
divergence_fragment_per_residue = Metric(
    name='divergence_fragment_per_residue', orm=None,
    description='The Jensen-Shannon divergence of interface residues from the position specific fragment profile',
    direction='min', function='rank', filter=True
)
divergence_evolution_per_residue = Metric(
    name='divergence_evolution_per_residue', orm=None,
    description='The Jensen-Shannon divergence of interface residues from the position specific evolutionary profile',
    direction='min', function='rank', filter=True
)
divergence_interface_per_residue = Metric(
    name='divergence_interface_per_residue', orm=None,
    description='The Jensen-Shannon divergence of interface residues from the typical interface background',
    direction='min', function='rank', filter=True
)
energy_distance_from_structure_background_mean = Metric(
    name='energy_distance_from_structure_background_mean', orm=None,
    description="The distance of the design's per residue energy from a design with no constraint on amino acid "
                "selection",
    direction='min', function='rank', filter=True
)
# START entity#_ metrics. These are used with numbers during actual metric acquisition
entity_c_terminal_helix = Metric(
    name='entity_c_terminal_helix', orm=ProteinMetadata.c_terminal_helix,
    description='Whether the entity has a c-terminal helix',
    direction='max', function='boolean', filter=True
)
entity_c_terminal_orientation = Metric(
    name='entity_c_terminal_orientation', orm=EntityMetrics.c_terminal_orientation,
    description='Whether the entity c-termini is closer to the assembly core or surface (1 is away, -1 is towards)',
    direction='max', function='rank', filter=True
)
entity_name = Metric(
    name='entity_name', orm=EntityMetrics.entity_id,
    description='The name of the entity',
    direction=None, function=None, filter=None
)
entity_interface_connectivity = Metric(
    name='entity_interface_connectivity', orm=DesignEntityMetrics.interface_connectivity,
    description='How embedded is the entity interface in the rest of the protein?',
    direction='max', function='normalize', filter=True
)
entity_interface_secondary_structure_fragment_topology = Metric(
    name='entity_interface_secondary_structure_fragment_topology', orm=EntityMetrics.interface_secondary_structure_fragment_topology,
    description='The Stride based secondary structure names of each unique element where possible values are - H:Alpha '
                'helix, G:3-10 helix, I:PI-helix, E:Extended conformation, B/b:Isolated bridge, T:Turn, C:Coil '
                '(none of the above)',
    #             'DSSP G:310 helix, H:α helix and I:π helix, B:beta bridge, E:strand/beta bulge, T:turns,'
    #             '     S:high curvature (where the angle between i-2, i, and i+2 is at least 70°) and'
    #             '     " "(space):loop'
    direction=None, function=None, filter=None
)
entity_interface_secondary_structure_topology = Metric(
    name='entity_interface_secondary_structure_topology', orm=EntityMetrics.interface_secondary_structure_topology,
    description='The Stride based secondary structure names of each unique element where possible '
                       'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                       'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
    direction=None, function=None, filter=None
)
entity_max_radius = Metric(
    name='entity_max_radius', orm=EntityMetrics.max_radius,
    description='The furthest point the entity reaches from the assembly core',
    direction='min', function='rank', filter=True
)
entity_max_radius_average_deviation = Metric(
    name='entity_max_radius_average_deviation', orm=PoseMetrics.entity_max_radius_average_deviation,
    description='In a multi-entity assembly, the total deviation of the maximum radii of each entity from one another',
    direction='min', function='rank', filter=True
)
# entity_max_radius_ratio_v = Metric(
# name='entity_max_radius_ratio_v', entity_max_radius_ratio_=None,
# description='The ratio of the maximum radius from a reference of component 1 versus 2',
#      direction=None, function=None, filter=None
# )
entity_min_radius = Metric(
    name='entity_min_radius', orm=EntityMetrics.min_radius,
    description='The closest point the entity approaches the assembly core',
    direction='max', function='rank', filter=True
)
entity_min_radius_average_deviation = Metric(
    name='entity_min_radius_average_deviation', orm=PoseMetrics.entity_min_radius_average_deviation,
    description='In a multi-entity assembly, the total deviation of the minimum radii of each entity from one another',
    direction='min', function='rank', filter=True
)
# entity_min_radius_ratio_v = Metric(
# name='entity_min_radius_ratio_v', orm=None,
# description='The ratio of the minimum radius from a reference of component 1 versus 2',
#      direction=None, function=None, filter=None
# )
entity_n_terminal_helix = Metric(
    name='entity_n_terminal_helix', orm=ProteinMetadata.n_terminal_helix,
    description='Whether the entity has a n-terminal helix',
    direction='max', function='boolean', filter=True
)
entity_n_terminal_orientation = Metric(
    name='entity_n_terminal_orientation', orm=EntityMetrics.n_terminal_orientation,
    description='Whether the entity n-termini is closer to the assembly core or surface (1 is away, -1 is towards)',
    direction='max', function='rank', filter=True
)
entity_number_mutations = Metric(
    name='entity_number_mutations', orm=DesignEntityMetrics.number_mutations,
    description='The number of mutations made',
    direction='min', function='rank', filter=True
)
entity_number_of_residues = Metric(
    name='entity_number_of_residues', orm=None,  # EntityMetrics.number_of_residues,
    description='The number of residues',
    direction='min', function='rank', filter=True
)
entity_number_of_residues_average_deviation = Metric(
    name='entity_number_of_residues_average_deviation', orm=None,
    # PoseMetrics.entity_number_of_residues_average_deviation,
    description='In a multi-entity assembly, the total deviation between all pairwise combinations of the '
                       'number of residues of each entity',
    direction='min', function='rank', filter=True
)
# entity_number_of_residues_ratio_v = Metric(
# name='entity_number_of_residues_ratio_v', description ',=None,='',
# =None, function=None, filter=None
# )
entity_percent_mutations = Metric(
    name='entity_percent_mutations', orm=DesignEntityMetrics.percent_mutations,
    description='The percentage of the entity that has been mutated',
    direction='min', function='rank', filter=True
)
entity_plddt = Metric(
    name='entity_plddt', orm=DesignEntityMetrics.plddt,
    description='From AlphaFold, whether the prediction at a residue (or average for a structure) is more '
                       'confident in the local distance difference when compared to nearby residue neighbors',
    direction='max', function='normalize', filter=True
)
entity_plddt_deviation = Metric(
    name='entity_plddt_deviation', orm=DesignEntityMetrics.plddt_deviation,
    description='The deviation from multiple measurements of entity_plddt',
    direction='min', function='normalize', filter=True
)
entity_predicted_aligned_error = Metric(
    name='entity_predicted_aligned_error', orm=DesignEntityMetrics.predicted_aligned_error,
    description='From AlphaFold, the mean of the predicted aligned error which indicates for every pair of '
                'residues, where the prediction is more confident in the relative position of two residues '
                '(lower better). PAE is more suitable than pLDDT for judging confidence in relative domain '
                'placements',
    direction='min', function='normalize', filter=True
)
entity_predicted_aligned_error_deviation = Metric(
    name='entity_predicted_aligned_error_deviation', orm=DesignEntityMetrics.predicted_aligned_error_deviation,
    description='The deviation from multiple measurements of entity_predicted_aligned_error',
    direction='min', function='normalize', filter=True
)
entity_predicted_aligned_error_interface = Metric(
    name='entity_predicted_aligned_error_interface', orm=DesignEntityMetrics.predicted_aligned_error_interface,
    description='From AlphaFold, the mean of the predicted aligned error values over a predicted interface',
    direction='min', function='normalize', filter=True
)
entity_predicted_aligned_error_interface_deviation = Metric(
    name='entity_predicted_aligned_error_interface_deviation', orm=DesignEntityMetrics.predicted_aligned_error_interface_deviation,
    description='The deviation from multiple measurements of entity_predicted_aligned_error_interface',
    direction='min', function='normalize', filter=True
)
# predicted_aligned_error_median = Metric(
# name='predicted_aligned_error_median',  # orm=', , Todo
# description='From AlphaFold, the median of the predicted aligned error values over a prediction',
#       direction='min', function='normalize', filter=True
# )
entity_predicted_interface_template_modeling_score = Metric(
    name='entity_predicted_interface_template_modeling_score', orm=DesignEntityMetrics.predicted_interface_template_modeling_score,
    description='This can serve for a visualisation of domain packing confidence within the interface, where a'
                       'value of 0 means most confident. See PMID:15476259',
    direction='max', function='normalize', filter=True
)
entity_predicted_interface_template_modeling_score_deviation = Metric(
    name='entity_predicted_interface_template_modeling_score_deviation', orm=DesignEntityMetrics.predicted_interface_template_modeling_score_deviation,
    description='The deviation from multiple measurements of entity_predicted_interface_template_modeling_score',
    direction='max', function='normalize', filter=True
)
entity_predicted_template_modeling_score = Metric(
    name='entity_predicted_template_modeling_score', orm=DesignEntityMetrics.predicted_template_modeling_score,
    description='This can serve for a visualisation of domain packing confidence within the structure, where a'
                'value of 0 means most confident. See PMID:15476259',
    direction='max', function='normalize', filter=True
)
entity_predicted_template_modeling_score_deviation = Metric(
    name='entity_predicted_template_modeling_score_deviation', orm=DesignEntityMetrics.predicted_template_modeling_score_deviation,
    description='The deviation from multiple measurements of entity_predicted_template_modeling_score',
    direction='max', function='normalize', filter=True
)
entity_radius = Metric(
    name='entity_radius', orm=EntityMetrics.radius,
    description='The center of mass of the entity from the assembly core',
    direction='min', function='rank', filter=True
)
entity_radius_average_deviation = Metric(
    name='entity_radius_average_deviation', orm=PoseMetrics.entity_radius_average_deviation,
    description='In a multi-entity assembly, the total deviation of the center of mass of each entity from one another',
    direction='min', function='rank', filter=True
)
# entity_radius_ratio_v = Metric(
#     name='entity_radius_ratio_v', orm=None,
#     description='',
#     direction=None, function=None, filter=None
# )
# entity_rmsd_oligomer = Metric(
#     name='entity_rmsd_oligomer', orm=None,
#     description='Root Mean Square Deviation of all CA atoms between the designed and predicted oligomer',
#     direction='min', function='normalize', filter=True
# )
entity_rmsd_prediction_ensemble_deviation = Metric(
    name='entity_rmsd_prediction_ensemble_deviation', orm=DesignEntityMetrics.rmsd_prediction_ensemble_deviation,
    description='The deviation in each measurement of the entity_rmsd_prediction_ensemble',
    direction='min', function='normalize', filter=True
)
entity_rmsd_prediction_ensemble = Metric(
    name='entity_rmsd_prediction_ensemble', orm=DesignEntityMetrics.rmsd_prediction_ensemble,
    description='The average Root Mean Square Deviation for all CA atoms from each predicted model',
    direction='min', function='normalize', filter=True
)
entity_symmetry_group = Metric(
    name='entity_symmetry_group', orm=ProteinMetadata.symmetry_group,
    description='The symmetry notation of the entity',
    direction=None, function=None, filter=True
)
entity_thermophilicity = Metric(
    name='entity_thermophilicity', orm=ProteinMetadata.thermophilicity,
    description='The extent to which the domains in the Entity are thermophilic',
    direction='max', function='boolean', filter=None
)
errat_accuracy = Metric(
    name='errat_accuracy', orm=None,
    description='The overall Errat score of the design',
    direction='max', function='rank', filter=True
)
errat_deviation = Metric(
    name='errat_deviation', orm=None,
    description='Whether a residue window deviates significantly from typical Errat distribution',
    direction='min', function='boolean', filter=True
)
evolution_constraint = Metric(
    name='evolution_constraint', orm=JobProtocol.evolution_constraint,
    description='Whether evolutionary constraints were used to constrain design',
    direction=None, function='boolean', filter=True
)
favor_residue_energy = Metric(
    name='favor_residue_energy', orm=DesignMetrics.favor_residue_energy,
    description='Total weight of sequence constraints used to favor certain amino acids in design. '
                'Only protocols with a favored profile have values',
    direction='max', function='normalize', filter=True
)
hydrophobicity = Metric(
    name='hydrophobicity', orm=DesignMetrics.hydrophobicity,
    description='The sum of all hydrophobicity values calculated from the collapse profile. See PMID:28507157',
    direction='min', function='rank', filter=True
)
initial_z_value = Metric(
    name='initial_z_value', orm=JobProtocol.initial_z_value,
    description='The z-value of the observed fragment pair root mean squared deviation (RMSD) compared to the '
                'fragment cluster RMSD that is required for an initial fragment match during docking',
    direction='min', function='normalize', filter=True
)
# Todo this metric seems wrong given the calculation of Bale 2016 versus others.
#  Hypothesis that the multiplicity isn't correct given the symmetry based skew
interaction_energy_complex = Metric(
    name='interaction_energy_complex', orm=DesignMetrics.interaction_energy_complex,
    description='The two-body (residue-pair) energy of the complex. No solvation energies',
    direction='min', function='rank', filter=True
)
interaction_energy_per_residue = Metric(
    name='interaction_energy_per_residue', orm=DesignMetrics.interaction_energy_per_residue,
    description='The two-body (residue-pair) energy of the complex on a per-residue basis. No solvation energies',
    direction='min', function='rank', filter=True
)
interface = Metric(
    name='interface', orm=JobProtocol.interface,
    description='True if only interface residues were designed',
    direction=None, function='boolean', filter=True
)
interface_area_hydrophobic = Metric(
    name='interface_area_hydrophobic', orm=DesignMetrics.interface_area_hydrophobic,
    description='Total hydrophobic interface buried surface area',
    direction='min', function='rank', filter=True
)
interface_area_polar = Metric(
    name='interface_area_polar', orm=DesignMetrics.interface_area_polar,
    description='Total polar interface buried surface area',
    direction='max', function='rank', filter=True
)
# Todo
#  Make a measure on a per-interface residue basis? This would prioritize continuous interfaces
interface_area_to_residue_surface_ratio = Metric(
    name='interface_area_to_residue_surface_ratio', orm=DesignMetrics.interface_area_to_residue_surface_ratio,
    description='The average ratio of interface buried surface area to the surface accessible residue area in '
                'the uncomplexed pose',
    direction='max', function='rank', filter=True
)
interface_area_total = Metric(
    name='interface_area_total', orm=DesignMetrics.interface_area_total,
    description='Total interface buried surface area',
    direction='max', function='rank', filter=True
)
interface_b_factor = Metric(
    name='interface_b_factor', orm=PoseMetrics.interface_b_factor,
    description='The average B-factor from each atom, from each interface residue',
    direction='max', function='rank', filter=True
)
interface_bound_activation_energy = Metric(
    name='interface_bound_activation_energy', orm=DesignMetrics.interface_bound_activation_energy,
    description='Energy required for the unbound interface to adopt the conformation in the complexed state',
    direction='min', function='rank', filter=True
)
interface_composition_similarity = Metric(
    name='interface_composition_similarity', orm=DesignMetrics.interface_composition_similarity,
    description='The similarity to the expected interface composition given interface buried surface '
                'area. 1 is similar to natural interfaces, 0 is dissimilar',
    direction='max', function='rank', filter=True
)
# These can represent both sides of an interface
interface_connectivity = Metric(
    name='interface_connectivity', orm=DesignEntityMetrics.interface_connectivity,
    description='How embedded is the total interface in the rest of the protein?',
    direction='max', function='normalize', filter=True
)
interface_energy = Metric(
    name='interface_energy', orm=DesignMetrics.interface_energy,
    description='DeltaG of the complexed and unbound (repacked) interfaces',
    direction='min', function='rank', filter=True
)
interface_energy_complex = Metric(
    name='interface_energy_complex', orm=DesignMetrics.interface_energy_complex,
    description='Total interface residue energy summed in the complexed state',
    direction='min', function='rank', filter=True
)
interface_energy_density = Metric(
    name='interface_energy_density', orm=DesignMetrics.interface_energy_density,
    description=f'Interface energy per interface {ANGSTROM}\N{SUPERSCRIPT TWO}. How much energy is achieved as a '
                'function of the given space?',
    direction='min', function='rank', filter=True
)
interface_energy_unbound = Metric(
    name='interface_energy_unbound', orm=DesignMetrics.interface_energy_unbound,
    description='Total interface residue energy summed in the unbound state',
    direction='min', function='rank', filter=True
)
interface_separation = Metric(
    name='interface_separation', orm=DesignMetrics.interface_separation,
    description='Median distance between all atom points on each side of the interface',
    direction='min', function='normalize', filter=True
)
interface_separation_core = Metric(
    name='interface_separation_core', orm=DesignMetrics.interface_separation_core,
    description='Median distance between all atom points on each side of the interface core positions',
    direction='min', function='normalize', filter=True
)
interface_separation_fragment = Metric(
    name='interface_separation_fragment', orm=DesignMetrics.interface_separation_fragment,
    description='Median distance between all atom points on each side of the interface fragment positions',
    direction='min', function='normalize', filter=True
)
interface_secondary_structure_count = Metric(
    name='interface_secondary_structure_count', orm=PoseMetrics.interface_secondary_structure_count,
    description='The number of unique secondary structures in the interface',
    direction='max', function='normalize', filter=True
)
interface_secondary_structure_fragment_count = Metric(
    name='interface_secondary_structure_fragment_count', orm=PoseMetrics.interface_secondary_structure_fragment_count,
    description='The number of unique fragment containing secondary structures in the interface',
    direction='max', function='normalize', filter=True
)
interface_secondary_structure_fragment_topology = Metric(
    name='interface_secondary_structure_fragment_topology',
    orm=PoseMetrics.interface_secondary_structure_fragment_topology,
    description='The Stride based secondary structure names of each unique element where possible '
                'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
    direction=None, function=None, filter=None
)
interface_secondary_structure_topology = Metric(
    name='interface_secondary_structure_topology',
    orm=PoseMetrics.interface_secondary_structure_topology,
    description='The Stride based secondary structure names of each unique element where possible '
                'values are - H:Alpha helix, G:3-10 helix, I:PI-helix, E:Extended conformation, '
                'B/b:Isolated bridge, T:Turn, C:Coil (none of the above)',
    direction=None, function=None, filter=None
)
# The above four metrics are valid for total, interface1, and interface2
interface_local_density = Metric(
    name='interface_local_density', orm=DesignMetrics.interface_local_density,
    description='A measure of the average number of atom neighbors for each atom in the interface',
    direction='max', function='rank', filter=True
)
interface_solvation_energy = Metric(
    name='interface_solvation_energy', orm=DesignMetrics.interface_solvation_energy,
    # free_energy of desolvation is positive for bound interfaces. unbound - complex
    description='The change in free energy resulting from hydration of the separated interface surfaces. '
                'Positive values indicate work is required to solvate surfaces upon dissociation',
    direction='min', function='rank', filter=True
)
interface_solvation_energy_activation = Metric(
    name='interface_solvation_energy_activation', orm=DesignMetrics.interface_solvation_energy_activation,
    # unbound - bound
    description='The change in solvation free energy resulting from minimizing the bound atomic configuration '
                'from the complex state, in the uncomplexed state. Positive values indicate work is required '
                'for solvating the minimized configuration',
    direction='min', function='rank', filter=True
)
interface_solvation_energy_bound = Metric(
    name='interface_solvation_energy_bound', orm=DesignMetrics.interface_solvation_energy_bound,
    description='The free energy of solvation terms in the separated interface surfaces. Positive values '
                'indicate unfavorable solvation energy terms from the atoms in the interface',
    direction='min', function='rank', filter=True
)
interface_solvation_energy_complex = Metric(
    name='interface_solvation_energy_complex', orm=DesignMetrics.interface_solvation_energy_complex,
    description='The free energy of solvation terms in the interface complex. Positive values indicate '
                'energy is required to solvate upon dissociation',
    direction='min', function='rank', filter=True
)
interface_solvation_energy_density = Metric(
    name='interface_solvation_energy_density', orm=DesignMetrics.interface_solvation_energy_density,
    description='The free energy resulting from hydration of the separated interface surfaces on a per '
                f'{ANGSTROM}\N{SUPERSCRIPT TWO} basis. Positive values indicate energy is required to solvate'
                ' upon dissociation',
    direction='max', function='rank', filter=True
)
interface_solvation_energy_unbound = Metric(
    name='interface_solvation_energy_unbound', orm=DesignMetrics.interface_solvation_energy_unbound,
    description='The free energy of solvation terms in the separate, repacked, interface surfaces. Positive'
                ' values indicate energy is required to solvate',
    direction='min', function='rank', filter=True
)
match_value = Metric(
    name='match_value', orm=JobProtocol.match_value,
    description='The value that is required for a high quality fragment match during docking',
    direction='max', function='normalize', filter=True
)
maximum_radius = Metric(
    name='maximum_radius', orm=PoseMetrics.maximum_radius,
    description='The maximum radius any entity extends from the assembly core',
    direction='min', function='rank', filter=True
)
minimum_matched = Metric(
    name='minimum_matched', orm=JobProtocol.minimum_matched,
    description='The number of matches required for a docked interface to pass',
    direction='max', function='normalize', filter=True
)
minimum_radius = Metric(
    name='minimum_radius', orm=PoseMetrics.minimum_radius,
    description='The minimum radius any entity approaches the assembly core',
    direction='max', function='rank', filter=True
)
multiple_fragment_ratio = Metric(
    name='multiple_fragment_ratio', orm=PoseMetrics.multiple_fragment_ratio,
    description='The extent that central fragment residues are represented by multiple fragment observations. '
                'Higher ratio means more fragment observations per residue, i.e. a value of 3 would mean on '
                'average, 3 fragments are observed to be overlapping every "central fragment residue" in the '
                'interface',
    direction='max', function='rank', filter=True
)
nanohedra_score = Metric(
    name='nanohedra_score', orm=PoseMetrics.nanohedra_score,
    description='Sum of the match scores (1 / 1 + Z-score^2) for all fragment residues weighted by 2^i '
                'where i is an observation in the set of all observations'
                'by their ranked match score. Maximum value of 2',
    direction='max', function='rank', filter=True
)
nanohedra_score_center = Metric(
    name='nanohedra_score_center', orm=PoseMetrics.nanohedra_score_center,
    description='The nanohedra_score for the center residue of a fragment observation. These residues are the'
                ' captain of their observed member fragments, are most likely involved in interactions with '
                'the member fragments, and therefore most closely align with their attributes',
    direction='max', function='rank', filter=True
)
nanohedra_score_center_normalized = Metric(
    name='nanohedra_score_center_normalized', orm=PoseMetrics.nanohedra_score_center_normalized,
    description='The central Nanohedra Score normalized by number of central fragment residues. The maximum value is 2',
    direction='max', function='rank', filter=True
)
nanohedra_score_normalized = Metric(
    name='nanohedra_score_normalized', orm=PoseMetrics.nanohedra_score_normalized,
    description='The Nanohedra Score normalized by number of fragment residues. The maximum value is 2',
    direction='max', function='rank', filter=True
)
neighbors = Metric(
    name='neighbors', orm=JobProtocol.neighbors,
    description='True if neighbors of design residues were designed',
    direction=None, function='boolean', filter=True
)
number_residues_interface_fragment_total = Metric(
    name='number_residues_interface_fragment_total', orm=PoseMetrics.number_residues_interface_fragment_total,
    description='The number of residues in the interface with fragment observationsfound',
    direction='max', function='rank', filter=True
)
number_residues_interface_fragment_center = Metric(
    name='number_residues_interface_fragment_center', orm=PoseMetrics.number_residues_interface_fragment_center,
    description='The number of interface residues that belong to a central fragment residue',
    direction='max', function='rank', filter=None
)
number_residues_design = Metric(
    name='number_residues_design', orm=DesignMetrics.number_residues_design,
    description='The number of residues selected for sequence design',
    direction='max', function='normalize', filter=True
)
number_residues_interface = Metric(
    name='number_residues_interface', orm=PoseMetrics.number_residues_interface,
    # Also DesignMetrics.number_residues_interface,
    description='The total number of interface residues found in the pose (default is residue CB within 8A)',
    direction='max', function='rank', filter=True
)
number_residues_interface_non_fragment = Metric(
    name='number_residues_interface_non_fragment', orm=PoseMetrics.number_residues_interface_non_fragment,
    description='The number of interface residues that are missing total fragment observations',
    direction='max', function='rank', filter=True
)
number_hbonds = Metric(
    name='number_hbonds', orm=DesignMetrics.number_hbonds,
    description='The number of residues making H-bonds in the total interface. Residues may make more than one H-bond',
    direction='max', function='rank', filter=True
)
number_fragments_interface = Metric(
    name='number_fragments_interface', orm=PoseMetrics.number_fragments_interface,
    description='The number of fragments found in the pose interface',
    direction='max', function='normalize', filter=True
)
number_mutations = Metric(
    name='number_mutations', orm=DesignMetrics.number_mutations,
    description='The number of mutations made to the pose (ie. wild-type residue to any other amino acid)',
    direction='min', function='normalize', filter=True
)
number_predictions = Metric(
    name='number_predictions', orm=JobProtocol.number_predictions,
    description='The number of predictions performed during the predict-structures module',
    direction='min', function='normalize', filter=True
)
observations = Metric(
    name='observations', orm=None,
    description='Number of unique design trajectories contributing to statistics',
    direction='max', function='rank', filter=True
)
observed_design = Metric(
    name='observed_design', orm=DesignMetrics.observed_design,
    description='Percent of designed residues found in the combined profile. 1 is 100%',
    direction='max', function='rank', filter=True
)
observed_design_mean_probability = Metric(
    name='observed_design_mean_probability', orm=None,
    description='The mean probability of the designed residues in the design profile. 1 is 100%',
    direction='max', function='rank', filter=True
)
observed_evolution = Metric(
    name='observed_evolution', orm=DesignMetrics.observed_evolution,
    description='Percent of designed residues found in the evolutionary profile. 1 is 100%',
    direction='max', function='rank', filter=True
)
observed_evolution_mean_probability = Metric(
    name='observed_evolution_mean_probability', orm=None,
    description='The mean probability of the designed residues in the evolutionary profile. 1 is 100%',
    direction='max', function='rank', filter=True
)
observed_fragment = Metric(
    name='observed_fragment', orm=DesignMetrics.observed_fragment,
    description='Percent of designed residues found in the fragment profile. 1 is 100%',
    direction='max', function='rank', filter=True
)
observed_fragment_mean_probability = Metric(
    name='observed_fragment_mean_probability', orm=None,
    description='The mean probability of the designed residues in the fragment profile. 1 is 100%',
    direction='max', function='rank', filter=True
)
observed_interface = Metric(
    name='observed_interface', orm=DesignMetrics.observed_interface,
    description='Percent of designed residues found in the interface fragment library. 1 is 100%',
    direction='max', function='rank', filter=True
)
observed_interface_mean_probability = Metric(
    name='observed_interface_mean_probability', orm=None,
    description='The mean probability of the interface residues in the interface fragment library. 1 is 100%',
    direction='max', function='rank', filter=True
)
percent_core = Metric(
    name='percent_core', orm=DesignMetrics.percent_core,
    description="The percentage of residues which are 'core' according to Levy, E. 2010",
    direction='max', function='normalize', filter=True
)
percent_fragment_coil = Metric(
    name='percent_fragment_coil', orm=PoseMetrics.percent_fragment_coil,
    description='The percentage of fragments represented from coiled SS elements',
    direction='max', function='normalize', filter=True
)
percent_fragment_helix = Metric(
    name='percent_fragment_helix', orm=PoseMetrics.percent_fragment_helix,
    description='The percentage of fragments represented from an a-helix SS elements',
    direction='max', function='normalize', filter=True
)
percent_fragment_strand = Metric(
    name='percent_fragment_strand', orm=PoseMetrics.percent_fragment_strand,
    description='The percentage of fragments represented from a b-strand SS elements',
    direction='max', function='normalize', filter=True
)
percent_interface_area_hydrophobic = Metric(
    name='percent_interface_area_hydrophobic', orm=DesignMetrics.percent_interface_area_hydrophobic,
    description='The percent of interface area which is occupied by hydrophobic atoms',
    direction='min', function='normalize', filter=True
)
percent_interface_area_polar = Metric(
    name='percent_interface_area_polar', orm=DesignMetrics.percent_interface_area_polar,
    description='The percent of interface area which is occupied by polar atoms',
    direction='min', function='normalize', filter=True
)
percent_mutations = Metric(
    name='percent_mutations', orm=DesignMetrics.percent_mutations,
    description='The percent of the design which has been mutated',
    direction='max', function='normalize', filter=True
)
percent_residues_fragment_interface_center = Metric(
    name='percent_residues_fragment_interface_center', orm=PoseMetrics.percent_residues_fragment_interface_center,
    description='The percentage of interface residues which are central fragment observations',
    direction='max', function='normalize', filter=True
)
percent_residues_fragment_interface_total = Metric(
    name='percent_residues_fragment_interface_total', orm=PoseMetrics.percent_residues_fragment_interface_total,
    description='The percentage of interface residues which are represented by fragment observations',
    direction='max', function='normalize', filter=True
)
percent_residues_non_fragment_interface = Metric(
    name='percent_residues_non_fragment_interface', orm=PoseMetrics.percent_residues_non_fragment_interface,
    description="The percentage of interface residues which aren't represented by fragment observations",
    direction='max', function='normalize', filter=True
)
percent_rim = Metric(
    name='percent_rim', orm=DesignMetrics.percent_rim,
    description="The percentage of residues which are 'rim' according to Levy, E. 2010",
    direction='min', function='normalize', filter=True
)
percent_support = Metric(
    name='percent_support', orm=DesignMetrics.percent_support,
    description="The percentage of residues which are 'support' according to Levy, E. 2010",
    direction='max', function='normalize', filter=True
)
plddt = Metric(
    name='plddt', orm=DesignMetrics.plddt,
    description='From AlphaFold, whether the prediction at a residue (or average for a structure) is more '
                'confident in the local distance difference when compared to nearby residue neighbors',
    direction='max', function='normalize', filter=True
)
plddt_deviation = Metric(
    name='plddt_deviation', orm=DesignMetrics.plddt_deviation,
    description='The deviation from multiple measurements of plddt',
    direction='min', function='normalize', filter=True
)
pose_length = Metric(
    name='pose_length', orm=PoseMetrics.pose_length,
    description='The total number of residues in the design',
    direction='min', function='rank', filter=True
)
pose_thermophilicity = Metric(
    name='pose_thermophilicity', orm=PoseMetrics.pose_thermophilicity,
    description='The extent to which the entities in the Pose are thermophilic',
    direction='max', function='rank', filter=True
)
predicted_aligned_error = Metric(
    name='predicted_aligned_error', orm=DesignMetrics.predicted_aligned_error,
    description='From AlphaFold, the mean of the predicted aligned error which indicates for every pair of '
                'residues, where the prediction is more confident in the relative position of two residues '
                '(lower better). PAE is more suitable than pLDDT for judging confidence in relative domain '
                'placements',
    direction='min', function='normalize', filter=True
)
predicted_aligned_error_deviation = Metric(
    name='predicted_aligned_error_deviation',
    orm=DesignMetrics.predicted_aligned_error_deviation,
    description='The deviation from multiple measurements of predicted_aligned_error',
    direction='min', function='normalize', filter=True
)
predicted_aligned_error_interface = Metric(
    name='predicted_aligned_error_interface',
    orm=DesignMetrics.predicted_aligned_error_interface,
    description='From AlphaFold, the mean of the predicted aligned error values over a predicted interface',
    direction='min', function='normalize', filter=True
)
predicted_aligned_error_interface_deviation = Metric(
    name='predicted_aligned_error_interface_deviation',
    orm=DesignMetrics.predicted_aligned_error_interface_deviation,
    description='The deviation from multiple measurements of predicted_aligned_error_interface',
    direction='min', function='normalize', filter=True
)
# predicted_aligned_error_median = Metric(
#     name='predicted_aligned_error_median',  # orm=', , Todo
#     description='From AlphaFold, the median of the predicted aligned error values over a prediction',
#     direction='min', function='normalize', filter=True
# )
predicted_interface_template_modeling_score = Metric(
    name='predicted_interface_template_modeling_score', orm=DesignMetrics.predicted_interface_template_modeling_score,
    description='This can serve for a visualisation of domain packing confidence within the interface, where a'
                'value of 0 means most confident. See PMID:15476259',
    direction='max', function='normalize', filter=True
)
predicted_interface_template_modeling_score_deviation = Metric(
    name='predicted_interface_template_modeling_score_deviation', orm=DesignMetrics.predicted_interface_template_modeling_score_deviation,
    description='The deviation from multiple measurements of predicted_interface_template_modeling_score',
    direction='max', function='normalize', filter=True
)
predicted_template_modeling_score = Metric(
    name='predicted_template_modeling_score', orm=DesignMetrics.predicted_template_modeling_score,
    description='This can serve for a visualisation of domain packing confidence within the structure, where a'
                'value of 0 means most confident. See PMID:15476259',
    direction='max', function='normalize', filter=True
)
predicted_template_modeling_score_deviation = Metric(
    name='predicted_template_modeling_score_deviation', orm=DesignMetrics.predicted_template_modeling_score_deviation,
    description='The deviation from multiple measurements of predicted_template_modeling_score',
    direction='max', function='normalize', filter=True
)
prediction_model = Metric(
    name='prediction_model', orm=JobProtocol.prediction_model,
    description='The name of the model used to perform predict-structures',
    direction=None, function='equals', filter=True
)
proteinmpnn_dock_cross_entropy_loss = Metric(
    name='proteinmpnn_dock_cross_entropy_loss', orm=PoseMetrics.proteinmpnn_dock_cross_entropy_loss,
    description='The total loss between ProteinMPNN probabilities in the unbound and complex states',
    direction='max', function='normalize', filter=True
)
proteinmpnn_dock_cross_entropy_per_residue = Metric(
    name='proteinmpnn_dock_cross_entropy_per_residue', orm=PoseMetrics.proteinmpnn_dock_cross_entropy_per_residue,
    description='The per-docked interface residue loss between ProteinMPNN probabilities in the unbound and '
                'complexed states',
    direction='max', function='normalize', filter=True
)
proteinmpnn_model_name = Metric(
    name='proteinmpnn_model_name', orm=JobProtocol.proteinmpnn_model_name,
    description='The name of the ProteinMPNN model used for design',
    direction=None, function='equals', filter=True
)
proteinmpnn_v_design_probability_cross_entropy_loss = Metric(
    name='proteinmpnn_v_design_probability_cross_entropy_loss', orm=PoseMetrics.proteinmpnn_v_design_probability_cross_entropy_loss,
    description='The total loss between the ProteinMPNN probabilities and the design profile probabilities',
    direction='min', function='normalize', filter=True
)
proteinmpnn_v_design_probability_cross_entropy_per_residue = Metric(
    name='proteinmpnn_v_design_probability_cross_entropy_per_residue', orm=PoseMetrics.proteinmpnn_v_design_probability_cross_entropy_per_residue,
    description='The per-docked interface residue cross entropy loss between the ProteinMPNN probabilities and'
                ' the design profile probabilities',
    direction='min', function='normalize', filter=True
)
proteinmpnn_v_evolution_probability_cross_entropy_loss = Metric(
    name='proteinmpnn_v_evolution_probability_cross_entropy_loss', orm=PoseMetrics.proteinmpnn_v_evolution_probability_cross_entropy_loss,
    description='The total loss between the ProteinMPNN probabilities and the evolution profile probabilities',
    direction='min', function='normalize', filter=True
)
proteinmpnn_v_evolution_probability_cross_entropy_per_residue = Metric(
    name='proteinmpnn_v_evolution_probability_cross_entropy_per_residue', orm=PoseMetrics.proteinmpnn_v_evolution_probability_cross_entropy_per_residue,
    description='The per-docked interface residue cross entropy loss between the ProteinMPNN probabilities and the '
                'evolution profile probabilities',
    direction='min', function='normalize', filter=True
)
proteinmpnn_v_fragment_probability_cross_entropy_loss = Metric(
    name='proteinmpnn_v_fragment_probability_cross_entropy_loss', orm=PoseMetrics.proteinmpnn_v_fragment_probability_cross_entropy_loss,
    description='The total loss between the ProteinMPNN probabilities and the fragment profile probabilities',
    direction='min', function='normalize', filter=True
)
proteinmpnn_v_fragment_probability_cross_entropy_per_residue = Metric(
    name='proteinmpnn_v_fragment_probability_cross_entropy_per_residue', orm=PoseMetrics.proteinmpnn_v_fragment_probability_cross_entropy_per_residue,
    description='The per-fragment interface residue cross entropy loss between the ProteinMPNN probabilities and the '
                'fragment profile probabilities',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_complex = Metric(
    name='proteinmpnn_score_complex', orm=DesignMetrics.proteinmpnn_score_complex,
    description='The per-residue average complex ProteinMPNN score',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_complex_per_designed_residue = Metric(
    name='proteinmpnn_score_complex_per_designed_residue',
    orm=DesignMetrics.proteinmpnn_score_complex_per_designed_residue,
    description='The average complex ProteinMPNN score for designed residues',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_complex_per_interface_residue = Metric(
    name='proteinmpnn_score_complex_per_interface_residue',
    orm=DesignMetrics.proteinmpnn_score_complex_per_interface_residue,
    description='The average complex ProteinMPNN score for interface residues',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_delta = Metric(
    name='proteinmpnn_score_delta', orm=DesignMetrics.proteinmpnn_score_delta,
    description='The per-residue average complex-unbound ProteinMPNN score',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_delta_per_designed_residue = Metric(
    name='proteinmpnn_score_delta_per_designed_residue',
    orm=DesignMetrics.proteinmpnn_score_delta_per_designed_residue,
    description='The difference between the average complex and unbound ProteinMPNN score for designed residues',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_delta_per_interface_residue = Metric(
    name='proteinmpnn_score_delta_per_interface_residue',
    orm=DesignMetrics.proteinmpnn_score_delta_per_interface_residue,
    description='The difference between the average complex and unbound ProteinMPNN score for interface residues',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_unbound = Metric(
    name='proteinmpnn_score_unbound', orm=DesignMetrics.proteinmpnn_score_unbound,
    description='The per-residue average unbound ProteinMPNN score',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_unbound_per_designed_residue = Metric(
    name='proteinmpnn_score_unbound_per_designed_residue',
    orm=DesignMetrics.proteinmpnn_score_unbound_per_designed_residue,
    description='The average unbound ProteinMPNN score for designed residues',
    direction='min', function='normalize', filter=True
)
proteinmpnn_score_unbound_per_interface_residue = Metric(
    name='proteinmpnn_score_unbound_per_interface_residue',
    orm=DesignMetrics.proteinmpnn_score_unbound_per_interface_residue,
    description='The average unbound ProteinMPNN score for interface residues',
    direction='min', function='normalize', filter=True
)
proteinmpnn_loss_complex = Metric(
    name='proteinmpnn_loss_complex', orm=DesignMetrics.proteinmpnn_loss_complex,
    description='The magnitude of information missing from the sequence in the complex state',
    direction='min', function='normalize', filter=True
)
proteinmpnn_loss_unbound = Metric(
    name='proteinmpnn_loss_unbound', orm=DesignMetrics.proteinmpnn_loss_unbound,
    description='The magnitude of information missing from the sequence in the unbound state',
    direction='min', function='normalize', filter=True
)
protocol = Metric(
    name='protocol', orm=DesignProtocol.protocol,
    description='Protocols utilized to search sequence space given fragment and/or evolutionary constraint information',
    direction=None, function=None, filter=False
)
# protocol_energy_distance_sum = Metric(
# name='protocol_energy_distance_sum', orm=None,
#     description='The distance between the average linearly embedded per residue energy co-variation '
#                 'between specified protocols. Larger = greater distance. A small distance indicates '
#                 'that different protocols arrived at the same per residue energy conclusions despite '
#                 'different pools of amino acids specified for sampling',
#     direction='min', function='rank', filter=True
# )
# protocol_similarity_sum = Metric(
# name='protocol_similarity_sum', orm=None,
#     description='The statistical similarity between all sampled protocols. Larger is more similar, '
#                 'indicating that different protocols have interface statistics that are similar '
#                 'despite different pools of amino acids specified for sampling',
#     direction='max', function='rank', filter=True
# )
# protocol_sequence_distance_sum = Metric(
# name='protocol_sequence_distance_sum', orm=None,
#     description='The distance between the average linearly embedded sequence differences between '
#                 'specified protocols. Larger = greater distance. A small distance indicates that '
#                 'different protocols arrived at the same per residue energy conclusions despite '
#                 'different pools of amino acids specified for sampling',
#     direction='min', function='rank', filter=True
# )
# psipred_match = Metric(
#     name='psipred_match', orm=,
#     description='fraction of residues that match the secondary structure assignment of PSIPRED',
#     direction='min', function='rank', filter=True
# )
# rama_prepro_per_res = Metric(
#     name='rama_prepro_per_res', orm=,
#     description='The mean per-residue ramachandran probability of amino acid before a proline.',
#     direction='min', function='rank', filter=True
# )
rim = Metric(
    name='rim', orm=DesignMetrics.rim,
    description="The number of 'rim' residues as classified by E. Levy 2010",
    direction='max', function='rank', filter=True
)
rmsd_complex = Metric(
    name='rmsd_complex', orm=DesignMetrics.rmsd_complex,
    description='Root Mean Square Deviation of all CA atoms between the refined (relaxed) and designed states',
    direction='min', function='normalize', filter=True
)
# rmsd_oligomer = Metric(
#     name='rmsd_oligomer', orm=,
#     description='Root Mean Square Deviation of all CA atoms between the designed and predicted oligomer',
#     direction='min', function='normalize', filter=True
# )
rmsd_prediction_ensemble_deviation = Metric(
    name='rmsd_prediction_ensemble_deviation',orm=DesignMetrics.rmsd_prediction_ensemble_deviation,
    description='The deviation in each measurement of the rmsd_prediction_ensemble',
    direction='min', function='normalize', filter=True
)
rmsd_prediction_ensemble = Metric(
    name='rmsd_prediction_ensemble', orm=DesignMetrics.rmsd_prediction_ensemble,
    description='The average Root Mean Square Deviation for all CA atoms from each predicted model',
    direction='min', function='normalize', filter=True
)
rosetta_reference_energy = Metric(
    name='rosetta_reference_energy', orm=DesignMetrics.rosetta_reference_energy,
    description='Rosetta Energy Term - A metric for the unfolded energy of the protein along with sequence fitting '
                'corrections',
    direction='max', function='rank', filter=True
)
sequence_loss_design = Metric(
    name='sequence_loss_design', orm=DesignMetrics.sequence_loss_design,
    description='The magnitude of information missing from the sequence compared to the design profile',
    direction='min', function='normalize', filter=True
)
sequence_loss_design_per_residue = Metric(
    name='sequence_loss_design_per_residue',orm=DesignMetrics.sequence_loss_design_per_residue,
    description='The magnitude of information missing from the sequence compared to the design profile on a per-residue'
                ' basis',
    direction='min', function='normalize', filter=True
)
sequence_loss_evolution = Metric(
    name='sequence_loss_evolution', orm=DesignMetrics.sequence_loss_evolution,
    description='The magnitude of information missing from the sequence compared to the evolution profile',
    direction='min', function='normalize', filter=True
)
sequence_loss_evolution_per_residue = Metric(
    name='sequence_loss_evolution_per_residue', orm=DesignMetrics.sequence_loss_evolution_per_residue,
    description='The magnitude of information missing from the sequence compared to the evolution profile on a '
                'per-residue basis',
    direction='min', function='normalize', filter=True
)
sequence_loss_fragment = Metric(
    name='sequence_loss_fragment', orm=DesignMetrics.sequence_loss_fragment,
    description='The magnitude of information missing from the sequence compared to the fragment profile',
    direction='min', function='normalize', filter=True
)
sequence_loss_fragment_per_residue = Metric(
    name='sequence_loss_fragment_per_residue', orm=DesignMetrics.sequence_loss_fragment_per_residue,
    description='The magnitude of information missing from the sequence compared to the fragment profile on a '
                'per-residue basis. Only counts residues with fragment information',
    direction='min', function='normalize', filter=True
)
sequence = Metric(
    name='sequence', orm=DesignMetrics.sequence,
    description='The amino acid sequence of the design',
    direction=None, function='equals', filter=False
)
shape_complementarity = Metric(
    name='shape_complementarity', orm=DesignMetrics.shape_complementarity,
    description='Measure of fit between two surfaces. See PMID:8263940',
    direction='max', function='normalize', filter=True
)
shape_complementarity_core = Metric(
    name='shape_complementarity_core', orm=DesignMetrics.shape_complementarity_core,
    description='Measure of fit between two surfaces at interface core positions',
    direction='max', function='normalize', filter=True
)
shape_complementarity_fragment = Metric(
    name='shape_complementarity_fragment', orm=DesignMetrics.shape_complementarity_fragment,
    description='Measure of fit between two surfaces at interface fragment positions',
    direction='max', function='normalize', filter=True
)
spatial_aggregation_propensity = Metric(
    name='spatial_aggregation_propensity', orm=DesignMetrics.spatial_aggregation_propensity,
    description='A measure of the aggregation propensity of exposed hydrophobic surface patches in the '
                'complexed state. Positive values are more aggregation prone, while negative values are less '
                'prone. See PMID:19571001',
    direction='min', function='normalize', filter=True
)
spatial_aggregation_propensity_interface = Metric(
    name='spatial_aggregation_propensity_interface', orm=DesignMetrics.spatial_aggregation_propensity_interface,
    description='A measure of the aggregation propensity of the interface. Positive values are more '
                'aggregation prone, while negative values are less prone. See PMID:19571001',
    direction='min', function='normalize', filter=True
)
spatial_aggregation_propensity_unbound = Metric(
    name='spatial_aggregation_propensity_unbound', orm=DesignMetrics.spatial_aggregation_propensity_unbound,
    description='A measure of the aggregation propensity of exposed hydrophobic surface patches in the unbound '
                'state. Positive values are more aggregation prone, while negative values are less prone. '
                'See PMID:19571001',
    direction='min', function='normalize', filter=True
)
support = Metric(
    name='support', orm=DesignMetrics.support,
    description="The number of 'support' residues as classified by E. Levy 2010",
    direction='max', function='rank', filter=True
)
sym_entry_number = Metric(
    name='sym_entry_number', orm=PoseMetadata.sym_entry_number,
    description='The particular SymEntry integer of the Pose',
    direction=None, function=None, filter=True
)
sym_entry_specification = Metric(
    name='sym_entry_specification', orm=PoseMetadata.sym_entry_specification,
    description='The particular SymEntry specification of the Pose',
    direction=None, function=None, filter=True
)
symmetric_interface = Metric(
    name='symmetric_interface', orm=PoseMetrics.symmetric_interface,
    description='Whether the Pose contains inter-entity interfaces that arise because of the global symmetry. '
                'These interfaces are not specified in the local symmetry of the Entity assembly alone',
    direction=None, function=None, filter=True
)
symmetry = Metric(
    name='symmetry', orm=PoseMetadata.symmetry,
    description='The resulting symmetry of the Pose',
    direction=None, function=None, filter=True
)
symmetry_dimension = Metric(
    name='symmetry_dimension', orm=PoseMetadata.symmetry_dimension,
    description='The underlying dimension of the design. 0 - point, 2 - layer, 3 - space group',
    direction=None, function=None, filter=True
)
temperature = Metric(
    name='temperature', orm=DesignProtocol.temperature,
    description='The temperature that the design was performed at. Bounded between (0-1]',
    direction='min', function='normalize', filter=True
)
term_constraint = Metric(
    name='term_constraint', orm=JobProtocol.term_constraint,
    description='Whether tertiary constraints were used to constrain design',
    direction=None, function='boolean', filter=True
)
# 'total_charge_per_res': dict()  # Todo
use_gpu_relax = Metric(
    name='use_gpu_relax', orm=JobProtocol.use_gpu_relax,
    description='Whether a GPU was used for AlphaFold prediction relaxation',
    direction=None, function='boolean', filter=True
)
# Rosetta based scores
REU = Metric(
    name='REU', orm=None,
    description='Rosetta Energy Units. Always 0. We can disregard',
    direction='min', function='rank', filter=True
)
hbonds_res_selection_unbound = Metric(
    name='hbonds_res_selection_unbound', orm=None,
    description='The specific h-bonds present in the bound pose',
    direction=None, function=None, filter=None
)
hbonds_res_selection_1_unbound = Metric(
    name='hbonds_res_selection_1_unbound', orm=None,
    description='The specific h-bonds present in the unbound interface1',
    direction=None, function=None, filter=None
)
hbonds_res_selection_2_unbound = Metric(
    name='hbonds_res_selection_2_unbound', orm=None,
    description='The specific h-bonds present in the unbound interface2',
    direction=None, function=None, filter=None
)
dslf_fa13 = Metric(
    name='dslf_fa13', orm=None,
    description='Rosetta Energy Term - disulfide bonding',
    direction=None, function=None, filter=None
)
fa_atr = Metric(
    name='fa_atr', orm=None,
    description='Rosetta Energy Term - lennard jones full atom atractive forces',
    direction=None, function=None, filter=None
)
fa_dun = Metric(
    name='fa_dun', orm=None,
    description='Rosetta Energy Term - Dunbrack rotamer library statistical probability',
    direction=None, function=None, filter=None
)
fa_elec = Metric(
    name='fa_elec', orm=None,
    description='Rosetta Energy Term - full atom electrostatic forces',
    direction=None, function=None, filter=None
)
fa_intra_rep = Metric(
    name='fa_intra_rep', orm=None,
    description='Rosetta Energy Term - lennard jones full atom intra-residue repulsive forces',
    direction=None, function=None, filter=None
)
fa_intra_sol_xover4 = Metric(
    name='fa_intra_sol_xover4', orm=None,
    description='Rosetta Energy Term - full atom intra-residue solvent forces',
    direction=None, function=None, filter=None
)
fa_rep = Metric(
    name='fa_rep', orm=None,
    description='Rosetta Energy Term - lennard jones full atom repulsive forces',
    direction=None, function=None, filter=None
)
fa_sol = Metric(
    name='fa_sol', orm=None,
    description='Rosetta Energy Term - full atom solvent forces',
    direction=None, function=None, filter=None
)
hbond_bb_sc = Metric(
    name='hbond_bb_sc', orm=None,
    description='Rosetta Energy Term - backbone/sidechain hydrogen bonding',
    direction=None, function=None, filter=None
)
hbond_lr_bb = Metric(
    name='hbond_lr_bb', orm=None,
    description='Rosetta Energy Term - long range backbone hydrogen bonding',
    direction=None, function=None, filter=None
)
hbond_sc = Metric(
    name='hbond_sc', orm=None,
    description='Rosetta Energy Term - side-chain hydrogen bonding',
    direction=None, function=None, filter=None
)
hbond_sr_bb = Metric(
    name='hbond_sr_bb', orm=None,
    description='Rosetta Energy Term - short range backbone hydrogen bonding',
    direction=None, function=None, filter=None
)
lk_ball_wtd = Metric(
    name='lk_ball_wtd', orm=None,
    description='Rosetta Energy Term - Lazaris-Karplus weighted anisotropic solvation energy?',
    direction=None, function=None, filter=None
)
omega = Metric(
    name='omega', orm=None,
    description='Rosetta Energy Term - Energy term for omega angle?',
    direction=None, function=None, filter=None
)
p_aa_pp = Metric(
    name='p_aa_pp', orm=None,
    description='"Rosetta Energy Term - statistical probability of an amino acid given angles phi',
    direction=None, function=None, filter=None
)
pro_close = Metric(
    name='pro_close', orm=None,
    description='Rosetta Energy Term - to favor closing of proline rings',
    direction=None, function=None, filter=None
)
rama_prepro = Metric(
    name='rama_prepro', orm=None,
    description='Rosetta Energy Term - amino acid dependent term to favor certain Ramachandran angles on residue '
                'before proline',
    direction=None, function=None, filter=None
)
yhh_planarity = Metric(
    name='yhh_planarity', orm=None,
    description='Rosetta Energy Term - favor planarity of tyrosine alcohol hydrogen',
    direction=None, function=None, filter=None)
# ]
