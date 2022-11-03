from symdesign.metrics import master_metrics, rosetta_required_metrics
from symdesign.utils import pretty_format_table
from symdesign.utils.path import term_constraint, evolution_constraint, hbnet, scout, program_command, \
    select_sequences, program_name, structure_background, current_energy_function, number_of_trajectories, force_flags,\
    interface_design, select_poses, analysis, protocol, select_designs, nanohedra, nano_publication, interface_metrics,\
    cluster_poses, optimize_designs

nltb = '\n\t'
module_help_string = f'"{program_command} MODULE --help" will direct you towards the proper formatting of program flags'
help_sentence = ' If you need guidance on structuring your job submission, include --help in any command.'
general_rosetta_module_flags = \
    f'If the script should be run multiple times, include the flag "--{number_of_trajectories}" INT. ' \
    f'If the specific flags should be generated fresh use "--{force_flags}"'
select_advice =\
    f'It\'s recommended that %s is performed after {select_poses} as typically, many of the designs ' \
    f'from a single pose will have similar characteristics. Given that the degrees of freedom between a pose and each' \
    f' of its designs are not too different if not rigid body degrees of freedom are sampled, the user would like to ' \
    f'prioritize features which pertain specifically to the particular sequence placed on the design.'
select_string = \
    f'It\'s intended that the user mix and match {select_poses}{select_designs} to curate their design hypothesis. ' \
    f'The outcome of these can be fed back into design steps by resubmitting the poses for further processing.'
select_poses_guide = \
    f'The purpose of {select_poses} is to take a pool of poses of interest and apply some sort of selection criteria' \
    f' to limit the pool to a more managable computational load. Selection proceed by applying a set of specified ' \
    f'filters and weights to the collected pose metrics. For instance, by filtering for poses based on four different' \
    f' metrics, say "interface_area_total > 1000, shape_complementarity > 0.65, ' \
    f'percent_residues_fragment_center > 0.3, and interface_composition_similarity > 0.6", those poses which contain' \
    f' metrics that satify each of these filters will be chosen. Additionally, by using three weights, ' \
    f'say "interface_secondary_structure_fragment_count = 0.33, pose_thermophilicity = 0.33, and ' \
    f'shape_complementarity 0.33" those passing designs can then be sorted according to their cumulative weight' \
    f' allowing you to prioritize poses and only push the best ones forward. In this way, you can take a large number' \
    f' of starting conformations and cull them to those that are more favorable given some set of design constraints.' \
    f' This protocol should be thought about carefully to ensure that metrics are utilized which capture pose ' \
    f'features and not sequence features. In the sequence feature case, select_sequences would be better suited' \
    f' to differentiate between sequences. {select_string}'
select_designs_sequences = \
    f'The purpose of {select_designs} is to analyze individual design trajectories (representing unique sequences)' \
    f' from your pool of poses. Selection proceeds by applying a set of specified filters and weights to the ' \
    f'collected design metrics. For instance, by filtering for design based on four different metrics, ' \
    f'say "percent_interface_area_polar > 0.35, shape_complementarity > 0.68, and errat_deviation < 1", those' \
    f' designs which contain metrics that satify each of these filters will be chosen. Additionally, by using four' \
    f' weights, say "interface_energy = 0.3, buried_unsatified_hbond_density = 0.3, interface_local_density = 0.2,' \
    f' and shape_complementarity 0.2" passing designs can then be sorted according to their cumulative weight' \
    f' allowing you to prioritize designs and only push the best ones forward. In this way, you can take a large' \
    f' number of designs and cull them to those that are more favorable given some set of design constraints. ' \
    f'{select_string}'
select_designs_guide = select_designs_sequences + select_advice % select_designs
select_sequences_guide = \
    select_designs_guide + (select_advice % select_sequences) + \
    f' When using {select_sequences}, options to output the selected designs as formatted protein/nucleotide' \
    f' sequences become available.'
pose_input_scalability = \
    'This is useful for instance, if you have a single pose and this pose had 10 designs output from it, you could' \
    ' issue interface_metrics to collect metrics for each of these designs. This logic holds for 10 poses with 10' \
    ' designs each, or 100 poses, etc.'
expand_asu_guide = f'Simply outputs each design as the fully expanded assembly from the ASU'
interface_metrics_guide = \
    f'Gathering accurate interface_metrics is the most critical part of the interface design process. As such, ' \
    f'{program_name} calculates interface_metrics after each design is processed from an interface_design job. This ' \
    f'module is simply a helper module which instructs the program to take metrics for any interface that was' \
    f' generated from a design step or any design protocol that you are particularly interested in. In particular the' \
    f' {interface_metrics} protocol describes the interface with a few key metrics that are collected from Rosetta ' \
    f'Filters and SimpleMetrics. These include necessary energetic measurements of the interface in the complexed, ' \
    f'the complexed atomic configuration yet removed from the bound complex, and an unbound state with atoms repacked' \
    f' as they adopt to a solvated state. If interface_metrics are not performed on your designs, or a pose is simply' \
    f' loaded for analysis, then there will be missing metrics. These include {", ".join(rosetta_required_metrics)},' \
    f' which are currently generated outside the {program_name} program environment. Many metrics such as surface' \
    f' area, secondary structure, and fragment descriptions, various entity sequence based metrics and parameters are' \
    f' always available for analysis.'
orient_guide = \
    'The purpose of orient is to take a symmetric system and place it into it\'s canonical symmetric orientation so' \
    ' that during the program operation, it is understood where the symmetry axes lie and thus how symmetry operators' \
    ' can be utilized to perform design calculations. Orient requires a symmetric system be input to run properly and' \
    ' doesn\'t need to be run after a symmetry is established with loaded designs. Therefore, an entire oligomer ' \
    'should be present in the submitted poses. Once the oligameric assembly is oriented an asymmetric unit will be ' \
    'extracted and saved alongside this assembly. This asymmetric unit will be the core of any future designs ' \
    'undertaken on the oriented design submission.'
cluster_poses_guide = \
    f'The {cluster_poses} module is useful for taking a specified pose input and clustering each according to a ' \
    f'desired metric to ensure that two poses in the set are not too similar. In the case that poses are similar or ' \
    f'identical, the most favorable one can be used while the others are overlooked. It is recommended to perform ' \
    f'{cluster_poses} before moving forward with a large set of candidate poses which may have some similarities ' \
    f'in structure or conformation. By combining redundancies, viable poses that are obscured can be revealed as well' \
    f' as narrowing down a set of 10 to 20 very similar poses.\n\tCurrently, there are three clustering metrics for ' \
    f'{cluster_poses}. The first is an interface alignment clustering algorithim carried out by the program iAlign. ' \
    f'iAlign takes an interface and measures how it\'s constituent CB atoms are configured, then matches that ' \
    f'configuration to other poses in the set. The second clustering methodology is to use a transformational cluster' \
    f'. Transformational clustering compares the transforms present in each symmetry group in a pose with other poses' \
    f' symmetry group transforms. This clustering method is a bit coarse, but is much more rapid and quickly finds ' \
    f'the best poses in fragment based {nanohedra.title()} docking. For instance, if two structural homologs are used in a' \
    f' set of poses, their similarity may be completely apparent to iAlign, but their entities are different ' \
    f'identities and therefore they wouldn\'t be equal by transformation. When thousands of poses are under ' \
    f'consideration, such as in the case of {nanohedra.title()} outputs, spatial transformation can quickly identify ' \
    f'matching poses.\n\tCluster maps are implemented through the argument -C and can be used in many cases ' \
    f'during a selection scheme such as in {select_poses}, {select_designs}, or {select_sequences}, This will allow a' \
    f' mapping from cluster representative to cluster members should be utilized to select the most favorable poses' \
    f' from those considered.'
interface_design_guide = \
    f'{interface_design} initializes all the necessary data and job files to create protein sequences capable of ' \
    f'forming a non-covalent interface between the components of interest, whether that interface be situated in a ' \
    f'symmetric system or not. The choice of amino acid at each interface residue position can be constrained ' \
    f'according to multiple requirements, all of which adopt the design philosophy that each individual residue ' \
    f'should be modeled in it\'s position specific context. As the most important factor in interface design is ' \
    f'whether the interface is capable of forming a bound complex, all modeling explores the prospects for a given ' \
    f'interface using the sequence design process, broadly understood as sampling random amino acid types at each ' \
    f'designable position while searching for the minima in a chosen energy function. Currently, the available ' \
    f'landscape is sampled and scored according to the Rosetta Energy Function (currently {current_energy_function}).' \
    f' To perform this search successfully and quickly, we constrain each residue\'s available amino acid types ' \
    f'according to the local environment. If no constraints are requested, pass the flag "--{structure_background}"' \
    f' and no sequence constraints will be used. Using evolutionarily allowed amino acids, defined as those amino ' \
    f'acids present in sequence homologs to the protein of interest, we ensure that sequence design favors the ' \
    f'production of soluble, folded proteins. As sequence homology captures the latent sequence information that ' \
    f'prescribes a protein\'s overall 3D fold, utilizing evolutionary constraint ensures the design generally ' \
    f'maintains its initial conformation upon mutation as during the interface design task. Evolutionary constraint ' \
    f'is utilized by default, however, providing the flag "--no-{evolution_constraint}" removes this constraint. ' \
    f'In essence, the evolutionarily observed frequencies guide the selection of amino acid types to sample during ' \
    f'design. As an additional constraint to form bound interfaces, position specific sequence design is augmented ' \
    f'with tertiary motifs (TERMs). TERMs represent structural motifs between a pair of naturally observed secondary ' \
    f'structures that confer a unique sequence-structure relationship favoring their geometric displacement and ' \
    f'interaction. As TERMs are placed across the interface, each additional TERM signifies a pair of surface residue' \
    f' orientations that are highly prevalent in natural protein motifs. By using associated TERM sequence ' \
    f'information, we can ensure the designed sequence reinforces the existing structure while ensuring interface ' \
    f'interactions. Practically speaking, TERM inclusion typically involves increasing the placement of hydrophobic ' \
    f'residues to form a tightly packed interface core. The inclusion of this sequence structure information ' \
    f'contrasts the typical surface environment of proteins and provides much needed interactions to favor tight ' \
    f'interfaces free of entropically disordered water. By default, design with TERMs is enabled, however, ' \
    f'providing the flag "--no-{term_constraint}" will disable this search. Finally, after specifying these position ' \
    f'specific amino acid constraints, we explicitly design hydrogen bond networks into the interface at applicable ' \
    f'positions and perform sequence design on the surrounding residues to promote the strongest interface possible. ' \
    f'By default, hydrogen bond network generation occurs, however, this can be skipped with "--no-{hbnet}". If the ' \
    f'design set up is favorable, it has the effect of utilizing hydrophobic residues to achieve tight binding, but ' \
    f'extensive hydrophobics are interrupted by polar networks which are intended to maintain the protein fold, ' \
    f'increase solubility, and provide interface specificity. Finally, to prevent excessive sampling and perform ' \
    f'rapid design on a small number of candidates to filter down to the most promising ones, provide the flag ' \
    f'"--{scout}" which greatly expedites interface design at a sacrifice for ultimate sampling depth and therefore ' \
    f'accuracy. It is not recommended to use any of the designs produced by the scout procedure, however. Instead, ' \
    f'they should be immediately followed by the analysis module to understand which out of a handful of input poses ' \
    f'are the most advantageous to the design task. After interface_design is completed, use the {analysis} module to' \
    f' prepare metrics for the modules {select_poses} and {select_sequences}. Importantly, {interface_design} can be ' \
    f'undertaken multiple times with different protocols to sample different characteristics of the interface. These ' \
    f'different protocols can be selected for in later modules using the flag --{protocol}.{help_sentence}'
formatted_metrics = \
    pretty_format_table([(metric, attributes['description']) for metric, attributes in sorted(master_metrics.items())])
analysis_guide = \
    f'After running "{program_command} {analysis}", the following metrics will be available for each pose ' \
    f'(unique design configuration) selected for analysis:\n\t{nltb.join(formatted_metrics)}\n\nAdditionally, you' \
    f' can view the pose specific files [pose-id]_Trajectory.csv for comparison of different design trials for an' \
    f' individual pose, and [pose-id]_Residues.csv for residue specific information over the various trajectories.' \
    f' Usage of overall pose metrics across all poses should facilitate selection of the best configurations to move' \
    f' forward with, while Trajectory and Residue information can inform your choice of sequence selection ' \
    f'parameters. Selection of the resulting poses/designs/sequences can be accomplished through the modules ' \
    f'"{select_poses}, {select_designs}, or {select_sequences}".\n\t{module_help_string}'
nanohedra_guide = \
    f'{nanohedra} operates the {nanohedra.title()} program which carries out the fragment based docking routine described in' \
    f' {nano_publication}. Fragment based docking is the main way {program_name} samples new symmetric materials' \
    f' from component building blocks, but can also be used to dock any two proteins based on surface fragment ' \
    f'complementarity or tertiary motifs (TERMs). To create new docked poses, protein surface patches are queried for' \
    f' TERM similarities using paired interface fragment libraries. ' \
    f'By finding the overlap of surface motifs from one component with the potential TERMs on ' \
    f' a second component, two protein components can be sampled for docking potential.'
# TODO
refine_guide = \
    ', as sometimes refinement, is useful for creating a more idealized structure to be put into subsequent design processes. However, refinement can also be a means of querying a model against a particular energy, landscape, and then pulling metrics out of that design, Therefore, all of these different processes rely on interface metrics '
optimize_designs_guide = \
    f'{optimize_designs} is used to fine-tune designs after identifying a viable set of designs. As the ' \
    f'interface design protocol is agnostic to the number of mutations generated, it is useful to ensure that any ' \
    f'designed sequence does not contain more mutations than required. {optimize_designs} can revert residue ' \
    f'identities back to wild type for residues that are not directly contributing to the pose energetics. ' \
    f'Additionally, designs may be optimized around additional goals, such as identifying poorly configured residues ' \
    f'in a design and specifying them for increased or altered degrees of freedom search.\n\tFor instance, if there ' \
    f'is a hydrophobic residue which is particularly ill-suited, you can specify that you would like to try a range ' \
    f'of non-hydrophobic residues at this site. This would test any polar amino acid which is ' \
    f'conserved in the design profile, but not hydrophobic and subject it to mutational scanning. If a particular ' \
    f'site is not indicated for any sort of specification, then the current design residue and the wild type residue' \
    f' will be the only modifications tested. If there are no detrimental effects from the calculations, including' \
    f' decreases in the shape_complementarity or an increase in the buried_unsatisfied_hbonds that are more ' \
    f'significant than a small percent, the reversion to wild type will be used. Additionally, small decreases in the' \
    f' interface_energy_density will be tolerated in an effort to reduce mutational burden.\n\t'
# Todo include these sentences when the module is capable
additional = \
    f'Additionally, {optimize_designs} can perform mutational steps to increase thermal stability consistent with the' \
    f' Protein Repair One Stop Shop (PROSS) protocol. In this case, the protocol uses the evolutionary profile to ' \
    f'mutate residue positions to highly favorable, evolutionarily acceptable, amino acid choices. {optimize_designs}' \
    f' can also include these positions into the refinement protocol, by using the flag, "stabilize". When used, the ' \
    f'stability of the designs at all non-interface locations may be modified to promote stability. In addition, the ' \
    f'flag "solubilize" will perform optimization by taking surface exposed hydrophobic residues and attempt to ' \
    f'mutate them polar residues, decreasing the overall hydrophobic burden. The final option "charge_match" will ' \
    f'perform mutagenesis of surface positions to match the surface to the correct overall charge density for a ' \
    f'specified organism. This may improve solubility by making the overall mobility within the cytosol equal to that' \
    f' of the organism typical cytosolic electrostatics.'
