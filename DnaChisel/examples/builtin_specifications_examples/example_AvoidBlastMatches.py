"""Example of use for AvoidBlastMatches.

In this example we create a 1000bp random sequence, then edit out every match
with E. coli that is 14bp or longer.

"""
import os
from genome_collector import GenomeCollection
from dnachisel import (
    DnaOptimizationProblem,
    random_dna_sequence,
    AvoidBlastMatches,
)

# THIS CREATES THE ECOLI BLAST DATABASE ON YOUR MACHINE IF NOT ALREADY HERE

collection = GenomeCollection()
ecoli_blastdb = collection.get_taxid_blastdb_path(511145, db_type="nucl")

# DEFINE AND SOLVE THE PROBLEM

problem = DnaOptimizationProblem(
    sequence=random_dna_sequence(500, seed=123),
    constraints=[
        AvoidBlastMatches(
            blast_db=ecoli_blastdb,
            min_align_length=13,
            perc_identity=100,
            word_size=5, # The bigger the word size, the faster
            e_value=1e20,
            # ungapped=False
        )
    ],
)

print(
    "Constraints validity before optimization\n",
    problem.constraints_text_summary(),
)

print("\nNow resolving the problems\n")
problem.resolve_constraints(final_check=True)

print(
    "Constraints validity after optimization\n",
    problem.constraints_text_summary(),
)
