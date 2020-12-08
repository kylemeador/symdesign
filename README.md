Align.py - Align terminal helices

AnalyzeMutatedSequences.py - Look at mutations to a pdb file compared to a wildtype

AnalyzeOutput.py - Analyze the metrics returned from a Rosetta design run

ASU.py - Design recap specific crap

Atom.py - The main class used in all of Josh's PDB manipulation

CommandDistributer.py - Used to submit large job arrays to cassini

ExpandASU.py - Expand origin centered cubic point group symmetries

GatherProfile.py - Retrieve the sequence profile for a pdb from hhblits

Model.py - Turn multiple PDB objects into a multimodel PDB file

NanohedraWrap.py - Generate commands to run Nanohedra on cassini

OrientOligomer.py - Orient a desired oligomer on the z-axis in a canonical setting

PDB.py - The main PDB object manipulation class

PoseProcessing.py - Set up a Nanohedra docked file for Rosetta 

ProteinExpression.py - Investigate a PDB's sequence tags to see if there should be one appended at a certain termini

QueryUniProtByPDBCode.py - find a uniprot ID for a PDB code

Residue.py - a super class of the Atom class

ScoreNative.py - Used to score 2 chain interfaces similarly to Nanohedra

SlurmControl.py - Random SLURM job handling stuff

Stride.py - Runs stride?

SymDesignControl - The master execution of the Nanohedra to Rosetta interface design to sequence selection pipeline for all SCM's. This is really what all the scripts above are set up to work best for.

SymDesignUtils.py - Tons of random functions used in above. Manipulating lists of PDB files, handling random files, generating position specific scoring matrices, multiple sequence alignments, handling fragment data, running multiprocessing, retrieving files from a directory, etc.

WriteSBATCH.py - Useful to write a sbtach script using your specific parameters