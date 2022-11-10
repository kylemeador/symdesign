# SymDesign
SymDesign is a command line interface used for hypothesis-driven protein-protein interface design. The package provides end to end support for the protein design process by seamlessly connecting multiple bioinformatics tools to create a rich data output describing the conformational space surrounding an interface design project. From docking, to design parameterization, to job distribution, and finally analysis of the resulting designs, SymDesign allows a single platform with organized data output to simplify the most complicated design pipelines. These tools allow a user to move from a design hypothesis to gene synthesis, with the only limitation being the conformational (computational) complexity of the chosen design process. 

In it's simplest usage, a protein structure with two entities sharing an interface (a design pose) is subjected to interface design using Rosetta's Metropolis criteria Monte Carlo sampling guided by the Rosetta score function. In it's most complicated use case, two or more entities in any symmetric assembly can be docked (only in permissible symmetries), have their interfaces queried for interface design prospects via evolutionary and fragment based statistical methods, and then be subjected to parallel processing for constrained interface re-design. After performing design calculations, a thorough set of interfacial measurements provides a rich overview of the characteristics present in your designs allowing for objective criteria to lead design selection. 

### Symmetry is at the core of the program.
Symmetry is used to simplify the investigation of possible interfaces between chains in a highly repetitive symmetric material and simplify calculations down to the fundamental connective unit, the interface. Previously, symmetric modelling has been a roadblock to sampling conformational space because of numerous complexities in enumerating interfacial possibilities when multiple components are involved in a symmetric assembly. This limitation has been overcome with the conformational sampling afforded by Nanohedra, an programmatic emphasis on single protein entities that combine to make up an asymmetric unit, and finally a thorough integration with Rosetta's underlying symmetry machinery. 

### Evolutionary augmented design
The second core feature, is that interface design is augmented through combining evolutionary information pertinent to each proteins unique tertiary topology with smaller protein fragments that collectively form tertiary motifs. The former provides short and long range selective pressures to the underlying stability of the protein fold, while the later allows augmentation of the interface with design patterns already highly sampled in nature. By mixing these two background sets of data, we can leverage all the data gleaned by the genomic and structural genomics explosions over the past two decades. Used as a selecting pressure on the physical-chemical energy functions, native like designs can be biased during sequence sampling and eventual design selection, simplifying biochemical characterization. 

Design proceeds with a few options, first and foremost, all residues in contact across an interface are designated designable and have their sidechain degrees of freedom sampled with harmonic constraints placed on the backbone. From this simple framework, multiple protocols for amino acid sampling are carried out to gather a distribution of the structural possibilities given different constraints. First, interface residue contacts can be layered with commonly observed tertiary structure motifs from the PDB, providing a measure of structural complementarity between the observed interface and natural tertiary motifs. Second, each residue can be constrained to sample only amino acids that are available given their protein's specific evolutionary context. Using both tertiary motifs and evolutionary constraint, a combination protocol can be specified which constrains design to only amino acids capable of satisfying both constraints. Each of these types of constraint can be measured against entirely free protocols where the choice of amino acid reflects only the free energy contributions guided by the score function. The key insight is to compare the constrained and free protocols. This allows us to understand how well suited an interface is for a low energy design given it's natural context. If the output from these design protocols can reach a consensus, it is incredibly likely that the confluence of data will provide improved mimicry of natural proteins and improved success of design pursuits.

To run SymDesign, prepare your design target either with seamless integration of Nanohedra docking `python SymDesign.py nanohedra` or use of another available docking software. Next, initialize your designs for processing `python SymDesign.py interface-design`, to create SBATCH scripts to then distribute to the computational cluster for parallel processing. Finally, when all design processing steps have finished, run analysis `python SymDesign.py analysis` to compile all the designs from a project into a single .csv dataframe. This dataframe can be viewed through Excel/Google Sheets, however, streamlined tools to analyze this data using the pandas module with Jupyter Notebooks for easy plotting and design analysis are in the works. Once you have performed analysis, you can specify metrics that you believe best represent your successful design targets by using `python SymDesign.py select-poses` and `python SymDesign.py select-sequences` to input parameters to fit the analysis for top ranking design poses and then top ranking sequence designs within that pose.

All of these modules come with a number of parameters that can modify the outcome. You can access the available options (flags) through `python SymDesign.py --help` or `python SymDesign.py MODULE --help` for module specific flags. 

As an example, you can specify whether you'd like evolutionary information `--evolutionary-constrint` or fragment information `--fragment-constraint` (both default) applied during design sampling. Any particular choice of entity/residue to include in the design can be specified as well. View these under the Design selectors arguments or the module `design-selectors`. (select, such as `--select-residues-`) or exclude (mask, such as `--mask-residues-`). Further you can specify if any residues or chains are required `--required-residues` beyond the interface (i.e. you want to redesign/stabilize a helical fusion that you're uncertain of). 

All flags can be provided to any module on the command line or by using the file specification notation `@` such as @my_favorite_flags.file in the specified command. Alternatively, these values will take their defaults if none are provided.

Some examples of viable commands:

    python SymDesign.py --directory path/to/DOCKING/OUTPUT interface-design --select-designable-chains A,B --mask-designable-residues-by-pose-number 243-287

Additionally, the interface metrics can be measured at any interface (symmetric and asymmetric)

    python SymDesign.py --project SymDesignOuput/Projects/D4_interface_analysis refine -symmetry D4:{D4} --gather-metrics

To use symmetry, specify a symmetry `--symmetry`/`--entry` during project initialization (or specify `--cryst` to use the CRYST1 record provided with a file) in the case of using 2D and 3D lattice symmetries inherent to a PDB entry for instance.

To turn an ASU into a full assembly, simply run
    
    python SymDesign.py -d designs/design_asus expand-asu --symmetry I:{C2}:{C5}

Where designs/design_asus is a directory containing files with an icosahedral asymmetric unit containing two chains, the C2 symmetry group first in every file, and the C5 symmetry group second.  

#### In order to use this set of tools, first you will need to set up your environment.  
Follow the instructions for this in the `python SymDesign.py --setup` output.

If you want to contribute, please feel free to reach out kylemeador@g.ucla.edu and I will invite you as a collaborator on github.com/kylemeador/symdesign.
