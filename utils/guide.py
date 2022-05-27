from SymDesignUtils import pretty_format_table
from PathUtils import no_term_constraint, no_evolution_constraint, no_hbnet, scout, program_command, select_sequences, \
    program_name, structure_background, current_energy_function, number_of_trajectories, force_flags, interface_design, \
    select_poses, analysis, protocol
from DesignMetrics import master_metrics, rosetta_required_metrics

help_sentence = ' If you need guidance on structuring your job submission, include --help in any command.'
general_rosetta_module_flags = \
    'If the script should be run multiple times, include the flag "--%s" INT. If the specific flags should be generated fresh use "--%s"' % (number_of_trajectories, force_flags)
select_poses_guide = \
    'The purpose of select poses is to take a pool of poses of interest and apply some sort of selection criteria to limit the pool to a more managable computational load. Selection proceed by applying a set of specified filters and weights to the collected pose metrics. For instance, by filtering for poses based on four different metrics, say "interface_area_total > 1000, shape_complementarity > 0.65, percent_residues_fragment_center > 0.3, and interface_composition_similarity > 0.6", those poses which contain metrics that satify each of these filters will be chosen. Additionally, by using three weights, say "interface_secondary_structure_fragment_count = 0.33, entity_thermophilicity = 0.33, and shape_complementarity 0.33" those passing designs can then be sorted according to their cumulative weight allowing you to prioritize poses and only push the best ones forward. In this way, you can take a large number of starting conformations and cull them to those that are more favorable given some set of design constraints. This protocol should be thought about carefully to ensure that metrics are utilized which capture pose features and not sequence features. In the sequence feature case, select_sequences would be better suited to differentiate between sequences.'
select_sequences_guide = \
    'The purpose of select sequences is to analyze the individual design trajectories (representing unique sequence designs) from your pool of poses. Selection proceeds by applying a set of specified filters and weights to the collected design metrics. For instance, by filtering for design based on four different metrics, say "percent_interface_area_polar > 0.35, shape_complementarity > 0.68, and errat_deviation < 0", those poses which contain metrics that satify each of these filters will be chosen. Additionally, by using three weights, say "interface_energy = 0.3, buried_unsatified_hbond_density = 0.3, interface_local_density = 0.2, and shape_complementarity 0.2" those passing designs can then be sorted according to their cumulative weight allowing you to prioritize designs and only push the best ones forward. In this way, you can take a large number of designs and cull them to those that are more favorable given some set of design constraints. It\'s recommended that select_sequences is performed after select_poses as typically, many of the sequences contained within a single pose will have similar characteristics. Given that the degrees of freedom between a pose and each of its designs are not too different if not rigid body degrees of freedom are sampled, the user would like to prioritize features which pertain specifically to the particular sequence placed on the design.'
select_string = \
    'It\'s intended that the user mix and match these two modules to curate their design hypothesis. The outcome of these can be fed back into design steps by resubmitting the poses for further processing.'
pose_input_scalability = \
    'This is useful for instance, if you have a single pose and this pose had 10 designs output from it, you could issue interface_metrics to collect metrics for each of these designs. This logic holds for 10 poses with 10 designs each, or 100 poses, etc.'
interface_metrics_guide = \
    'Gathering accurate interface_metrics is the most critical part of the interface design process. As such, %s calculates interface_metrics after each design is processed from an interface_design job. This module is simply a helper module which instructs the program to take metrics for any interface that was generated from a design step or any design protocol that you are particularly interested in. In particular the interface_metrics protocol describes the interface with a few key metrics that are collected from Rosetta Filters and SimpleMetrics. These include necessary energetic measurements of the interface in the complexed, the complexed atomic configuration yet removed from the bound complex, and an unbound state with atoms repacked as they adopt to a solvated state. If interface_metrics are not performed on your designs, or a pose is simply loaded for analysis, then there will be missing metrics. These include %s which are currently generated outside the %s program environment. Many metrics such as surface area, secondary structure, and fragment descriptions, various entity sequence based metrics and parameters are always available for analysis.' % (program_name, ', '.join(rosetta_required_metrics), program_name)
orient_guide = \
    'The purpose of orient is to take a symmetric system and place it into it\'s canonical symmetric orientation so that during the program operation, it is understood where the symmetry axes lie and thus how symmetry operators can be utilized to perform design calculations. Orient requires a symmetric system be input to run properly and doesn\'t need to be run after a symmetry is established with loaded designs. Therefore, an entire oligomer should be present in the submitted poses. Once the oligameric assembly is oriented an asymmetric unit will be extracted and saved alongside this assembly. This asymmetric unit will be the core of any future designs undertaken on the oriented design submission.'
cluster_poses_guide = \
    'The cluster_poses module is useful for taking a specified pose input and clustering them according to some metric to ensure that two poses are not too similar. In the case that poses are similar or identical, the most favorable one can be kept while the others discarded. Currently, there are two enable clustering metrics for cluster_poses. The first is an interface alignment clustering algorithim carried out by the program iAlign. iAlign takes an interface and measures how it\'s constituent Cb atoms are configured, then matches that configuration to other poses. The second clustering of pose methodology is to use a transformational cluster. Transformational clustering compares the transforms present in each entity in a pose with other poses entity transforms. This transformational cluster is a bit coarse, but is much more rapid. For instance, if two structural homologs are used then their similarity may be apparent, but their entities are different identities and therefore they wouldn\'t be compared. When thousands of poses are under consideration, such as in the case of fragment based Nanohedra docking outputs, spatial transformation can quickly identify matching entity pairs. It is recommended to perform pose clustering before moving forward with a large set of candidate poses, which may have some overlapping similarities. Cluster_poses therefore weeds out redundancies that may be obscuring viable poses, which wouldn\'t selected due to some calculation number constraint. sample other configurations in the case where there are multiple poses around a very ideal situation, as well as help find those configurations that are the most ideal given a set of 10 to 20 possible. Ones surrounding a local configurational space. Cluster maps are implemented through the argument -C and can be used in many cases during a selection scheme such as in select poses or select sequences, This will allow a mapped template to show how the clustering should be utilized to then select the most favorable sequences from those filtered.'
interface_design_guide = \
    f'{interface_design} initializes all the necessary data and job files to create protein sequences capable of forming a non-covalent interface between the components of interest, whether that interface be situated in a symmetric system or not. The choice of amino acid at each interface residue position can be constrained according to multiple requirements, all of which adopt the design philosophy that each individual residue should be modeled in it\'s position specific context. As the most important factor in interface design is whether the interface is capable of forming a bound complex, all modeling explores the prospects for a given interface using the sequence design process, broadly understood as sampling random amino acid types at each designable position while searching for the minima in a chosen energy function. Currently, the available landscape is sampled and scored according to the Rosetta Energy Function (currently {current_energy_function}). To perform this search successfully and quickly, we constrain each residue\'s available amino acid types according to the local environment. If no constraints are requested, pass the flag "--{structure_background}" and no sequence constraints will be used. Using evolutionarily allowed amino acids, defined as those amino acids present in sequence homologs to the protein of interest, we ensure that sequence design favors the production of soluble, folded proteins. As sequence homology captures the latent sequence information that prescribes a protein\'s overall 3D fold, utilizing evolutionary constraint ensures the design generally maintains its initial conformation upon mutation as during the interface design task. Evolutionary constraint is utilized by default, however, providing the flag "--{no_evolution_constraint}" removes this constraint. In essence, the evolutionarily observed frequencies guide the selection of amino acid types to sample during design. As an additional constraint to form bound interfaces, position specific sequence design is augmented with tertiary motifs (TERMs). TERMs represent structural motifs between a pair of naturally observed secondary structures that confer a unique sequence-structure relationship favoring their geometric displacement and interaction. As TERMs are placed across the interface, each additional TERM signifies a pair of surface residue orientations that are highly prevalent in natural protein motifs. By using associated TERM sequence information, we can ensure the designed sequence reinforces the existing structure while ensuring interface interactions. Practically speaking, TERM inclusion typically involves increasing the placement of hydrophobic residues to form a tightly packed interface core. The inclusion of this sequence structure information contrasts the typical surface environment of proteins and provides much needed interactions to favor tight interfaces free of entropically disordered water. By default, design with TERMs is enabled, however, providing the flag "--{no_term_constraint}" will disable this search. Finally, after specifying these position specific amino acid constraints, we explicitly design hydrogen bond networks into the interface at applicable positions and perform sequence design on the surrounding residues to promote the strongest interface possible. By default, hydrogen bond network generation occurs, however, this can be skipped with "--{no_hbnet}". If the design set up is favorable, it has the effect of utilizing hydrophobic residues to achieve tight binding, but extensive hydrophobics are interrupted by polar networks which are intended to maintain the protein fold, increase solubility, and provide interface specificity. Finally, to prevent excessive sampling and perform rapid design on a small number of candidates to filter down to the most promising ones, provide the flag "--{scout}" which greatly expedites interface design at a sacrifice for ultimate sampling depth and therefore accuracy. It is not recommended to use any of the designs produced by the scout procedure, however. Instead, they should be immediately followed by the analysis module to understand which out of a handful of input poses are the most advantageous to the design task. After interface_design is completed, use the {analysis} module to prepare metrics for the modules {select_poses} and {select_sequences}. Importantly, {interface_design} can be undertaken multiple times with different protocols to sample different characteristics of the interface. These different protocols can be selected for in later modules using the flag --{protocol}.{help_sentence}'
formatted_metrics = \
    pretty_format_table([(metric, attributes['description']) for metric, attributes in sorted(master_metrics.items())])
analysis_guide = \
    'After running \'%s analysis\', the following metrics will be available for each pose (unique design configuration) selected for analysis:\n\t%s\n\nAdditionally, you can view the pose specific files [pose-id]_Trajectory.csv for comparison of different design trials for an individual pose, and [pose-id]_Residues.csv for residue specific information over the various trajectories. Usage of overall pose metrics across all poses should facilitate selection of the best configurations to move forwardwith, while Trajectory and Residue information can inform your choice of sequence selection parameters. Selection of the resulting poses can be accomplished through the %s module \'%s\'.\n\t\'%s %s -h\' will get you started.' % (program_command, '\n\t'.join(formatted_metrics), program_name, select_sequences, program_command, select_sequences)

# TODO
refine_guide = \
    ', as sometimes refinement, is useful for creating a more idealized structure to be put into subsequent design processes. However, refinement can also be a means of querying a model against a particular energy, landscape, and then pulling metrics out of that design, Therefore, all of these different processes rely on interface metrics '
optimize_designs_guide = \
    'The optimized design module is used to fine-tune is after rounds of interface design metrics collection, and filtering have identified, a viable set of designs, particularly as the task of interface design is agnostic, to the number of mutations generated. It is often favorable to ensure that any design sequence does not contain more mutations than required. This is where the optimized designs package can be utilized to create, reversions back to wild type sequence that are not directly contributing to the pose energetics, as well as to, particularly optimize a set of designs around. Local goals. Such as identifying poorly configured. Residues in a design and specifying them for higher resolution degrees of freedom searching. For instance, if there is a hydrophobic residue which is particularly ill-suited one could specify that you would like to try the range of non-hydrophobic residues at this particular site. This would then generate any sort of polar residues which are non-considered, not hydrophobic and subject them to the optimized designs protocol. If a particular site is not indicated for any sort of specification, then the current design residue and the wild type rescue as specified. From the input structure are the only two that will be queried taking then the wild type residue. If there are no detrimental effects from the calculations detrimental effects include decreases in the shape complementarity that are more significant than a small percentage of the overall shape, comparatory and increase in the unsatisfied, hydrogen bonding potential and sharp decreases into the interface energy density. As such these all work to ensure that the mutational burden is balanced with the interface design, objective. Additionally. The optimized designs protocol can perform what would be considerably known as increased thermal stability. For instance, the program protein repair, one-stop shop or pros, Generally uses a multiple sequence alignment and this evolutionary profile to find, highly favorable, evolutionarily acceptable, amino acid, mutations optimized designs. Can also include these positions into the refinement protocol, where in the flag, stabilize can be supplied The stability of the design at all locations. That are not in the interface, maybe be sure it up by using the most evolutionarily favored. Amino acids In addition, the flag solubilize can be issued which would then take any surface exposed hydrophobic residues and attempt to mutate them. To surface exposed surface accessible polar residues, Therefore decreasing the overall, hydrophobic burden of a particular molecule and replacing it non-interface residues with polar residues, Therefore compensating for a particularly hydrophobic patch, placed upon a new interface. The final option charge match will specify that the surface is of the correct. Overall charge density for their requested organism. This may in improve the solubility by making the overall mobility within the cytosol or in line with that of the organism will Constraints.'

# interface_design
# 'Outcome of this and design program.
# With a interface design submission, there are three key processes that are underplay. First is the specification of any evolutionary constraints that may contribute to the interface design outcome. These evolutionary constraints are captured from multiple sequence alignments that are run within the program and saved applying the same alignment across all entities used in various poses.
# The second core functionality of interface design is to generate fragments across interfaces. These fragments are instances of secondary structures that are highly observed in the known protein database. And as such their interface, observances will indicate whether a particular tertiary structure is prone to favoring. A design outcome by gathering these metrics.
# It can be associated that the sequence and structure relationships that exist within these tertiary motifs, can then be applied to the design. These are mixed with evolutionary constraints in order to create some designs constraints and the form of a design profile indicating. Which amino acids, frequencies are likely to be useful and contribute to the overall design goal, This can help balance the trade-offs between creating an interface from a surface that may have existed as a charged, and polar surface patch to one that is now required to be hydrophobic and complementary to a nonexisting partner that is also specified to be complementary and put possibly The third and final important to ask in the enter.
# Today's design process is using all of these constraints generated to specify where the interface lies. The symmetry of how it lies and then perform the Monte Carlo search across this interface for the possible amino acid choices and their energetic favorability. Now, this search can be specified using constraints of fragment or evolutionary and other search methods can be incorporated, which prioritize hydrogen bonding, or which may utilize either of the previous constraints and not the other or no.
# Constraints at all to find the most ideal structural form, that is accessible without any constraints as interface design. Module adapts, new protocols for the design may become available. Particularly, if you have a specific design case, incorporation of a new design protocol into the interface design, is easily accessible by preparing a Rosetta script and ensuring that the script submission is in line with standards and design operating schemes.'
# interface_metrics_guide = ''
# #
