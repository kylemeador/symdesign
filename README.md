# SymDesign
SymDesign is a command line interface (CLI) used for hypothesis-driven protein-protein interface design. The package provides end to end support for the protein design process by seamlessly connecting multiple bioinformatics tools to create a rich data output describing the conformational space surrounding an interface design project. From docking, to design parameterization, to job distribution, and finally analysis of the resulting designs, SymDesign allows a single platform with organized data output to simplify the most complicated design pipelines. These tools allow a user to move from a design hypothesis to gene synthesis, with the only limitation being the conformational (computational) complexity of the chosen design process. 

In it's simplest usage, a protein structure with two entities sharing an interface (a design pose) is subjected to interface design using Rosetta's Metropolis criteria Monte Carlo sampling guided by the Rosetta score function. In it's most complicated use case, two or more entities in any symmetric assembly can be docked (only in permissible symmetries), have their interfaces queried for interface design prospects via evolutionary and fragment based statistical methods, and then be subjected to parallel processing on cassini's many computational nodes for constrained interface re-design. After performing design calculations, a thorough set of interfacial measurements provides a rich overview of the characteristics present in your designs allowing for objective criteria to lead design selection. 

### Symmetry is at the core of the program.
Symmetry is used to simplify the investigation of possible interfaces between chains in a highly repetitive symmetric material and simplify calculations down to the fundamental connective unit, the interface. Previously, symmetric modelling has been a roadblock to sampling conformational space because of numerous complexities in enumerating interfacial possibilities when multiple components are involved in a symmetric assembly. This limitation has been overcome with the conformational sampling afforded by Nanohedra, an programmatic emphasis on single protein entities that combine to make up an asymmetric unit, and finally a thorough integration with Rosetta's underlying symmetry machinery. 

### Evolutionary augmented design
The second core feature, is that interface design is augmented through combining evolutionary information pertinent to each proteins unique tertiary topology with smaller protein fragments that collectively form tertiary motifs. The former provides short and long range selective pressures to the underlying stability of the protein fold, while the later allows augmentation of the interface with design patterns already highly sampled in nature. By mixing these two background sets of data, we can leverage all the data gleaned by the genomic and structural genomics explosions over the past two decades. Used as a selecting pressure on the physical-chemical energy functions, native like designs can be biased during sequence sampling and eventual design selection, simplifying biochemical characterization. 

Design proceeds with a few options, first and foremost, all residues in contact across an interface are designated designable and have their sidechain degrees of freedom sampled with harmonic constraints placed on the backbone. From this simple framework, multiple protocols for amino acid sampling are carried out to gather a distribution of the structural possibilities given different constraints. First, interface residue contacts can be layered with commonly observed tertiary structure motifs from the PDB, providing a measure of structural complementarity between the observed interface and natural tertiary motifs. Second, each residue can be constrained to sample only amino acids that are available given their protein's specific evolutionary context. Using both tertiary motifs and evolutionary constraint, a combination protocol can be specified which constrains design to only amino acids capable of satisfying both constraints. Each of these types of constraint can be measured against entirely free protocols where the choice of amino acid reflects only the free energy contributions guided by the score function. The key insight is to compare the constrained and free protocols. This allows us to understand how well suited an interface is for a low energy design given it's natural context. If the output from these design protocols can reach a consensus, it is incredibly likely that the confluence of data will provide improved mimicry of natural proteins and improved success of design pursuits.

To run SymDesign, prepare your design target either with seamless integration of Nanohedra docking `python SymDesign.py nanohedra` or use of another available docking software. Next, initialize your designs for processing `python SymDesign.py interface_design`, to create SBATCH scripts to then distribute to the computational cluster for parallel processing. Finally, when all design processing steps have finished, run analysis `python SymDesign.py analysis` to compile all the designs from a project into a single .csv dataframe. This dataframe can be viewed through Excel/Google Sheets, however, streamlined tools to analyze this data using the pandas module with Jupyter Notebooks for easy plotting and design analysis are in the works. Once you have performed analysis, you can specify metrics that you believe best represent your successful design targets by using `python SymDesign.py select_designs` and `python SymDesign.py select_sequences` to input parameters to fit the analysis for top ranking design poses and then top ranking sequence designs within that pose.

All of these modules come with a number of parameters that can modify the outcome. You can access the available options (flags) through `python SymDesign.py flags`. Using this module, a CLI can help you generate design flags to guide each step in the process above. As an example, for design, you can specify whether you'd like evolutionary information or fragment information applied to the sampling as well as any specific entities, chains, or residues you would like to focus on (select) or exclude (mask) from design. Further you can specify if any residues or chains are required in design beyond the interface design (say you want to correct a helical fusion that your uncertain of the best overlapping sequence). All of these can be provided to any module by using the notation *@*my_favorite_flags.file in the specified command. Alternatively, these values will take their defaults if none are provided or if you only have one flag that your really interested in you can simply add this to the command.  
Some examples of viable commands:

    python SymDesign.py --directory DOCKING/OUTPUT design --symmetry T33 --nanohedra_output --no_term_constraint --select_designable_chains A,B --mask_designable_residues_by_pose_number 243-287

To measure the fragment propensities at the interface of the entities in the structure, run

    python SymDesign.py -d path/to/DOCKING/OUTPUT generate_fragments

Additionally, the fragment propensities can be measured at the interface of symmetric entities by specifying a symmetry (or providing a CRYST1 record in the .pdb file) in the case of 2D and 3D symmetries

    python SymDesign.py -d path/to/DOCKING/OUTPUT generate_fragments -symmetry T:{C2}:{C3}

To turn an ASU into a full assembly, simply run
    
    python SymDesign.py -d path/to/DOCKING/OUTPUT expand_asu -symmetry I:{C2}:{C5}

#### In order to use this set of tools, first you will need to set up your environment on cassini.  
I have done this using the SymDesignEnvironment.yaml provided to initialize the SymDesign environment in conda. If you are using anaconda/conda (which I recommend), `conda env create --file SymDesignEnvironment.yaml` will handle this for you. If you are using something else, there is probably an easy way to ensure your virtual environment is up to speed with SymDesign's dependencies.  
Next, you must add the following variable to your .bashrc (or .tschrc) so that the hhblits dependency can be correctly sourced. 

    export PATH=/home/kmeador/symdesign/dependencies/hh-suite/build/bin:$PATH
    or on .tcshrc
    setenv PATH /home/kmeador/symdesign/dependencies/hh-suite/build/bin:$PATH
    
    Additionally, add this path if you want to build your own scripts with any of the modules for easy import into python
    export PYTHONPATH=/yeates1/kmeador/symdesign:$PYTHONPATH
    or
    setenv PYTHONPATH /yeates1/kmeador/symdesign:$PYTHONPATH

If you want to contribute, please feel free to reach out kylemeador@g.ucla.edu and I will invite you as a collaborator on github.
