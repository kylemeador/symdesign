# symdesign
symdesign is python package for hypothesis-driven symmetric protein design. The package provides end to end support for the protein design process by seamlessly connecting multiple bioinformatics tools to create a rich data output describing the conformational space surrounding a design project. The code was born out of protein-protein interface design and offers extensive features for protein docking, sequence design, and finally, analysis of the resulting designs, especially for systems of symmetric proteins. Beyond protein docking, symdesign serves as a platform to perform interface and design analysis on existing structures as well as sampling new sequences or scoring existing ones with ProteinMPNN, and folding sequences into structures with AlphaFold. The integration of each of these tools in a single environment allow a user to move from hypothesis to sampling to analysis and finally sequence formatting for gene synthesis. Current limitations include the creation of new backbones, such as with generative backbone models. We are always looking for new ideas to extend the existing project features so don't be shy in generating a [pull request](https://github.com/kylemeador/symdesign/pulls) or starting a [discussion](https://github.com/kylemeador/symdesign/discussions).

## Installation
1. First, you will need to set up your python environment. After cloning this repository, run `conda env create --file path/to/symdesign/conda_env.yml`. Might I also suggest using [mamba](https://github.com/mamba-org/mamba) inplace of conda for an even faster experience. This command will create a suitable environment to access all the contained tools. Apple users of M1 and later chips will need different dependencies (WIP).
2. Execute `sudo apt-get install build-essential autoconf libc++-dev libc++abi-dev` to install system dependencies for [FreeSASA](https://github.com/mittinatten/freesasa#compatibility-and-dependencies). There may be different dependencies if you are utilizing an operating system other than ubuntu.
3. Follow the instructions for final set up and initialization of dependencies using the script `python path/to/symdesign/setup.py`

## Symmetry
Symmetry is used to simplify the investigation of different protein systems and scales. Whether there is local symmetry in an oligomer, or global symmetry, between multiple oligomers, each type of possible interface contact can be parsed apart leading to simpler analysis of a single gene from a highly repetitive symmetric material. Previously, symmetric modelling has been a roadblock to sampling conformational space because of numerous complexities in enumerating possibilities when multiple components are involved in a symmetric assembly. We have adapted the core symmetric principles from the Nanohedra project, placed a programmatic emphasis on single protein entities that combine to make up an asymmetric unit, and finally a thorough integration with sequence design methodologies. See `python path/to/symdesign symmetry --help` or `python path/to/symdesign symmetry --guide` to see the options for specifying symmetry as well as a guide to its usage. A great place to start exploring symmetry is `python path/to/symdesign symmetry --query result`.

To use symmetry, specify `--symmetry`/`--sym-entry` during project initialization. Also the keyword 'cryst' as in `--symmetry cryst` indicates the program should use the CRYST1 record provided with along with a file such as in the case of using 2D and 3D lattice symmetries.

## Design methodologies
The default design methodology is ProteinMPNN. There is also the possibility to integrate Rosetta based design if a license for Rosetta is acquired. To initialize usage with Rosetta, run `python symdesign/setup.py --rosetta`

### Ethos for evolution guided design
A core feature of all design and analysis is the integration of the design and analysis processes with profile information. Here, we define a design profile as the per-residue amino acid distribution describing the sequence-structure relationship space of a target design. A profile can be created from evolutionary information present from multiple sequence alignment based queries, termed an 'evolutionary_profile', which provide pertinent single and paired residue constraints to establish the protein tertiary topology. A profile can also be created from smaller protein fragments, a 'fragment_profile' that collectively form tertiary motifs. These have primarily used to parameterize protein interfaces, however, they are generated from sampling pairs of observed secondary structures that are fundamental to protein folding, and as such can describe nearly any tertiary structure. Whereas evolutionary_profile information provides short and long range selective pressures to the underlying stability of the protein fold, the fragment_profile in it's typically usage allows augmentation of interfaces with sequence-structure relationship patterns statistically favored in natural protein systems. By mixing these two background sets of data, in design or data analysis we can leverage all the data gleaned by the genomic and structural genomics explosions over the past two decades to help design outcomes.  

Additionally, the profiles calculated by ProteinMPNN can be accessed. In basic `design`, the ProteinMPNN 'inference_profile' is used to score sequences it creates. However, the 'structure_profile' can also be assessed by asking ProteinMPNN for a profile conditioned on the backbone coordinates.  

### Design protocols
Design proceeds with a few options, first and foremost, which residues. As an example of tailored selection, any particular choice of entity/residue can be included in design such as the usage of `--design` or exclusion of residues such as `--mask` residues. See the 'python path/to/symdesign design-residues --guide' for more information on setting these up. Additionally, preconfigured selections based on interfaces can be used. At a minimum residues in contact across an interface are designed with `--interface`. Additionally, neighbors of interface residues can be added to the design using `--neighbors`. Each of these naive distance based metrics can be configured with the inputs `--interface-distance`/`--neighbor-distance`. Additionally, you can specify whether you'd like evolutionary information `--evolutionary-constraint` or fragment information `--fragment-constraint` applied during design sampling.  

From this simple framework, multiple protocols for amino acid sampling can be carried out to gather a distribution of the structural possibilities given different constraints. For example, `--interface` residue contacts can be designed and subject to `--fragment-constraint` which layers the interface with commonly observed tertiary structure motifs from the PDB in the fragment_profile. This could indicate a measure of structural complementarity between the observed interface and natural tertiary motifs. Additionally, every residue in the protein `design` (module default) could use `--evolutionary-constraint` to sample only amino acids that are available given the protein's evolutionary_profile. Using both `--fragment-constraint` and `--evolutionary-constraint`, a combination protocol can be specified which constrains design to only amino acids capable of satisfying both constraints. Insight can be gained by comparing different protocols and understanding how well various information is capable of being represented in the design space. Such analysis has allowed us to understand how well suited an interface is for a low energy design given it's natural context [Meador et al. 2023 in preparation](www.idontwork.com). If the output from these design protocols can reach a consensus, it is increasingly likely that generated designs more easily minic natural proteins which can result in improved biochemical success.  

## Your first run
In it's simplest usage, a protein structure with a single protein entity can be analyzed, modeled for missing density, or redesigned sequences. In the most complicated use case, two or more entities in an infinite symmetric assembly can be docked in permissible symmetries, have the de novo interface and backbones subjected to sequence design with various constraints, and finally, fold those sequences into different homomeric or heteromeric systems. After performing any docking, design, or folding calculations, a thorough set of measurements, most focused on interfaces, however easily parameterized with other calculations can generate a rich overview of the design wise, or per residue positions characteristics present in the chosen poses/designs. The goal is to enable hypotheses to lead to design discovery through objective analysis of the design space.  

To run symdesign, prepare a design target either with seamless integration of Nanohedra docking `python path/to/symdesign nanohedra` HelixDisplay `python path/to/symdesign align-helices` or use of another available backbone generation program. Next, sample sequences for your coordinate space through `python path/to/symdesign design`. When any design processing steps occur, analysis is automatically performed. To retrieve specific structures or sequences and their corresponding analytic data, it is easy to implement data quality filters using the select tools such as `python path/to/symdesign select-*` where * can be one of `poses`, `designs`, or `sequences`, i.e. `python path/to/symdesign select-designs --filter`. Additionally, the corresponding project directories can be accessed to view any design files (SymDesignOutput/Projects/my-new-octahedron/). All data from a select-* protocol is formatted in .csv files which can easily be viewed through Excel/Google Sheets, however, we also have integrated tools to analyze and plot data using tools as IPython Notebooks to perform plotting and analysis with pandas, seaborn, and matplotlib. Finally, if designs are deemed desireable for design in the lab, the module `python path/to/symdesign select-sequences` simplifies formatting of nucleotide sequences for subsequent ordering.  

All of these modules come with a number of parameters that can modify the outcome. You can access the available options through `python path/to/symdesign --help` or `python path/to/symdesign MODULE --help` for module specific options.  

All flags can be provided to any module on the command line or by using the file specification notation `@` such as `@my_favorite_flags.file` in the specified command. Alternatively, these values will take their defaults if none are provided.  

#### Some examples of viable commands:

    python path/to/symdesign --directory path/to/other/OUTPUT design --design-residues A:243-287
    python path/to/symdesign --directory path/to/other/OUTPUT design --design-residues A:243-287 --symmetry C3

##### metrics can be measured between interfaces in biologically relevant oligomeric units (including monomers)

    python path/to/symdesign --project SymDesignOutput/Projects/D4_interface_analysis refine --measure-pose
    python path/to/symdesign --project SymDesignOutput/Projects/D4_interface_analysis refine --measure-pose --symmetry D4:{D4}{C1}

##### To turn an ASU into a full assembly, simply run
    
    python path/to/symdesign --directory designs/design_asus expand-asu --symmetry I:{C2}:{C5}

Where designs/design_asus is a directory containing files with an icosahedral asymmetric unit containing two chains, the C2 symmetry group first in every file, and the C5 symmetry group second.

## Scale
symdesign was designed to access different scales for any design workflow. Whether this is on a personal laptop, a Google Colab, or a computational cluster, the goal is for you to focus on the design task and seemlessly utilize the computational power given the available  resources. Of course, there are always limits to how much can be utilized at one time, especially given symmetry and GPU space. That being said, as many minimizations as possible have been made to perform design calculations even on infinite materials. 
### Relational database
The key requirement to scale as your workflow does is the database backend. Out of the box, symdesign ships with sqlite which maintains a relational database between all design inputs and outputs. If large scale parallel processing is desired, a custom database should probably be created and access specified with `--database-url`. Such access only needs to be configured once for a root symdesign directory (ex. SymDesignOutput). This cautionary note is given as sqlite can run into latency at extremely high loads. Importantly, concurrent database access is unavoidably impossible if your on an NFS file system. If you are operating on one, I suggest you find an alternative if concurrency is desired. symdesign sets up it's own database resources, but you will need to enable correct permissions to create tables, read, write, update and likely other database specific actions. If you are curious in setting up a custom database, please start a pull request or a discussion about the type of access you are creating and we can ensure that the sqlalchemy does its magic.

# Acknowledgements
symdesign would not be possible without the likes of many fantastic projects. The most integral projects include:
* [numpy](https://github.com/numpy/numpy)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [pandas](https://github.com/pandas-dev/pandas])
* [sqlalchemy](https://github.com/sqlalchemy/sqlalchemy)
* [Nanohedra](https://github.com/nanohedra/nanohedra)
* [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
* [AlphaFold](https://github.com/deepmind/alphafold)
* [FreeSASA](https://github.com/mittinatten/freesasa)
* [hhsuite](https://github.com/soedinglab/hh-suite)
* [Stride](https://webclu.bio.wzw.tum.de/stride/)
* [DNAChisel](https://github.com/Edinburgh-Genome-Foundry/DnaChisel)
* [conda](https://github.com/conda/conda)

## Contributing
If you want to contribute, please feel free to reach out kylemeador@g.ucla.edu and I will invite you as a collaborator on github.com/kylemeador/symdesign.
