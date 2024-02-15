"""
#### Provide objects for manipulation and modeling of protein structures

This module allows you to load protein structural files, manipulate their
 coordinates, and measure aspects of their position, relationships, and
 shape properties. Additionally, identify structural symmetry and inherently
model symmetric relationships, including their interfaces.

## [structure][]

* An [Atom][structure.base.Atom] has a [Coordinate][structure.coordinates.Coordinates].
```
|Atom|                                                         <- Structure
 |xyz|                                                         <- Coordinate
```
* A [Residue][structure.base.Residue] [ContainsAtoms][structure.base.ContainsAtoms] and [ContainsCoordinates][structure.base.StructureBase].
```
|Residue - - - - - - |                                         <- Structure
 [Atom|Atom|Atom|Atom]                                         <- Structure
```
* A [Structure][structure.model.Structure] [ContainsResidues][structure.base.ContainsResidues].
```
|Structure - - - - - - - - - - - - - - - - - - - - - - - - - | <- Structure
 [Residue|Residue|Residue|Residue|Residue|Residue|Residue|...] <- Structure
```
* A [Structure][structure.model.Structure] also [ContainsAtoms][structure.base.ContainsAtoms] and [ContainsCoordinates][structure.base.StructureBase].
```
|Structure - - - - - - - - - - - - - - - - - - - - - - - - - | <- Structure
 [Atom|Atom|Atom|Atom|Atom|Atom|Atom|Atom|Atom|Atom|Atom|Atom] <- Structure
 [xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz|xyz] <- Coordinate
```
* A [Chain][structure.model.Chain] is an instance of a [Structure][structure.model.Structure].
```
|Chain - - - - - - - - - - - - - - - - - - - - - - - - - - - | <- Structure
```
* A [Complex][structure.model.Complex] is a [Structure][structure.model.Structure] that [ContainsChains][structure.model.ContainsChains].
```
|Complex - - - - - - - - - - - - - - - - - - - - - - - - - - | <- Structure
/Structure * - * - * - * - * - * - * - * - * - * - * - * - * \ <- Structure
 [Chain- - - - - - - - - - - - |Chain- - - - - - - - - - - - ] <- Structure
```

### Loading a [Structure][structure.model.Structure]

## [sequence][]

* Every [Structure][structure.model.Structure] is a [GeneProduct][structure.model.StructuredGeneEntity].
```
|Structure - - - - - - - - - - - - - - - - - - - - - - - - - | <- Structure
/GeneProduct-> MPEAIRELNGHILFNCKALVDTGSSYPKQCDAKTGMIALQRPESA \\ <- Sequence(Protein)
```
* Each [GeneProduct][structure.model.StructuredGeneEntity] represents a [Gene][structure.model.GeneEntity].
```
|GeneProduct - - - - - - - - - - - - - - - - - - - - - - - - | <- Sequence
/Gene-> ATGCCTGAAGCTATTCGTGAATTAAATGGTCATATTTTATTTAATTGTAAAG \\ <- Sequence(DNA)
/ CTTTAGTTGATACTGGTTCTTCTTATCCTAAACAATGTGATGCTAAAACTGGTATGAT \\
/ TGCTTTACAACGTCCTGAATCTGCT - - - -  - - - - - - - - - - - - \\
```
* An [Entity][structure.model.Entity] maps a biological instance of a [Structure][structure.model.Structure] which [ContainsChains][structure.model.ContainsChains](1-N) and is a [Complex][structure.model.Complex], to a single [GeneProduct][structure.model.StructuredGeneEntity].
```
|Entity- - - - - - - - - - - - - - - - - - - - - - - - - - - | <- Structure
/GeneProduct * - * - * - * - * - * - * - * - * - * - * - * - \ <- Seq
 ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
 [Chain- - - - - - - - - - - - - - - - - - - - - - - - - - - ] <- Structure
 /GeneProduct- - - - - - - - - - - - - - - - - - - - - - - - \ <- Seq
 [Chain- - - - - - - - - - - - |Chain- - - - - - - - - - - - ] <- Structure
 /GeneProduct- - - - - - - - - |GeneProduct- - - - - - - - - \ <- Seq
 [Chain- - - - - - - |Chain- - - - - - - |Chain- - - - - - - ] <- Structure
 /GeneProduct- - - - |GeneProduct- - - - |GeneProduct- - - - \ <- Seq
 [Chain- - - - -|Chain - - - - |Chain- - - - -|Chain - - - - ] <- Structure
 /GeneProduct- -|GeneProduct - |GeneProduct- -|GeneProduct - \ <- Seq
 [Chain- - |Chain- - |Chain- - |Chain- - |Chain- - |Chain- - | <- Structure
 /GeneProd.|GeneProd.|GeneProd.|GeneProd.|GeneProd.|GeneProd.\ <- Seq
```
* As a result of GeneProduct duplication, an [Entity][structure.model.Entity] can be a [SymmetricModel][structure.model.SymmetricModel].

### Working with interfaces
* A Complex [ContainsGeneProducts][structure.model.StructuredGeneEntity].
```
|Complex- - - - - - - - - - - - - - - - - - - - - - - - - - -| <- Structure
 [Entity- - - - - - - - - - - - - - - - - - - - - - - - - - -] <- Seq
 [Entity- - - - - - - - -||Entity- - - - - - - - - - - - - - ] <- Seq
```

### Working with pieces of [Structure][structure.model.Structure]

* A [Fragment][structure.fragment.Fragment] [ContainsResidues][structure.base.ContainsResidues], but only a few and thus is a small [Structure][structure.model.Structure] representation.
```
|Fragment - - - - - - - - - - - - - - - -|                     <- Structure
 [Residue|Residue|Residue|Residue|Residue]                     <- Structure
```
#### Each [Fragment][structure.fragment.Fragment] can map to other [Fragment][structure.fragment.Fragment] instances so that larger [Structure][structure.model.Structure] can be broken down into pairs of [Fragment][structure.fragment.Fragment]. Such [Fragment][structure.fragment.Fragment] overlap with neighboring [Fragment][structure.fragment.Fragment] such as those that aare mapped to a separate "sliding window" register.
```
|Structure - - - - - - - - - - - - - - - - - - - - - - - - - | <- Structure
 Fragment- - - |Fragment - - -|Fragment- - - |Fragment - - - | <- Structure
    |Fragment- - - |Fragment - - -|Fragment- - - |             <- Structure
        |Fragment- - - |Fragment - - -|Fragment- - - |         <- Structure
            |Fragment- - - |Fragment - - -|Fragment- - - |     <- Structure
```


The module contains the following classes:

- `Pose()` - Create a symmetry-aware object to manipulate a collection of Entity instances
- `Model()` - Create an object to manipulate a collection of Chain instances
- `Entity()` - Create a symmetry-aware object to manipulate structurally homologous Chain instances

The module contains the following functions:

- `()` - Returns

Examples:
    >>> from pathlib import Path
    >>> from symdesign import structure
    >>>
    >>> structure_file = Path(Path.cwd(), 'tests', '1xyz.pdb1')
    >>> # structure_file = Path(Path.cwd(), 'tests', '1xyz-assembly1.cif')
    >>> pose_1xyz = structure.model.Pose.from_file(structure_file)
    >>> # Transform the pose to the coordinate origin
    >>> pose_1xyz.transform(translation=-pose_1xyz.center_of_mass)
    >>> chain_a_1xyz = pose_1xyz.get_chain('A')
    >>> for residue in chain_a_1xyz.residues:
    ...     print(f'Residue {residue.type}{residue.number} has a B-factor of {residue.b_factor} and {residue.sasa} '
    ...           '$\AA$\N{SUPERSCRIPT TWO} of solvent accessible surface area')
    Residue M1 has a B-factor of 47.08 and 34.02 A2 of solvent accessible surface area
    Residue V2 has a B-factor of 54.11 and 12.83 A2 of solvent accessible surface area
    ...
    >>>
    >>> chain_a_1xyz_per_residue_sasa = sum([residue.sasa for residue in chain_a_1xyz.residues])
    >>> chain_a_1xyz_per_residue_sasa == chain_a_1xyz.sasa
    True
    >>> for entity_idx, entity in enumerate(pose_1xyz.entities, 1):
    ...     print(f'The Entity #{entity_idx} is {entity.name}')
    ...     print(entity == chain_a_1xyz)

Structure initialization arguments
Process various types of Structure containers to update the Model with the corresponding information

Args:
    -*Passed to ContainsEntities*-
    entities: bool | list[Entity] | Structures = True - Whether to create Entity instances from passed Structure
        container instances, or existing Entity instances to create the Model with
    entity_names: Sequence = None - Names explicitly passed for the Entity instances. Length must equal number
        of entities. Names will take precedence over query_by_sequence if passed

    -*Passed to ContainsChains*-
    chains: bool | list[Chain] | Structures = True - Whether to create Chain instances from passed Structure
        container instances, or existing Chain instances to create the Model with
    query_by_sequence: bool = True - Whether the PDB API should be queried for an Entity name by matching
        sequence. Only used if entity_names not provided

    -*Passed to ContainsResidues*-
    structure: ContainsResidues = None - Create the instance based on an existing Structure instance
    fragment_db: FragmentDatabase = None - The identity of the FragmentDatabase to use for Fragment
        based operations
    residues: list[Residue] | Residues = None - The Residue instances which should constitute a new
        instance
    residue_indices: list[int] = None - The indices which specify the particular Residue instances to make this
        ContainsResidues instance. Used with a parent to specify a subdivision of a larger structure

    -*Passed to ContainsAtoms*-
    atoms: list[Atom] | Atoms = None - The Atom instances which should constitute a new Structure instance

    -*Passed to StructureBase*-
    parent: StructureBase = None - If another Structure object created this Structure instance, pass the
        'parent' instance. Will take ownership over Structure containers (coords, atoms, residues) for
        dependent Structures
    log: Log | Logger | bool = True - The Log or Logger instance, or the name for the logger to handle parent
        Structure logging. None or False prevents logging while any True assignment enables it
    coords: Coords | np.ndarray | list[list[float]] = None - When setting up a parent Structure instance, the
        coordinates of that Structure
    name: str = None - The identifier for the Structure instance

    -*Passed to StructureMetadata*-
    biological_assembly: str | None = None - The integer of the biological assembly
        (as indicated by PDB AssemblyID format)
    cryst_record: str | None - The string specifying how the molecule is situated in a lattice
    entity_info: dict[str, dict[dict | list | str]] - A mapping of the metadata to their distinct molecular
        identifiers
    file_path: AnyStr | None - The location on disk where the file was accessed
    reference_sequence: str | dict[str, str] = None - The reference sequence according to expression sequence or
        reference database
    resolution: float | None = None: The level of detail available from an experimental dataset contributing to
        the sharpness with which structural data can contribute towards building a model
"""
from . import utils, coordinates, fragment, sequence
from . import base
from . import model
from .model import Entity, Model, Pose, Structure