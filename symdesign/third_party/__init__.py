import os.path
import sys

# import alphafold
# import freesasa
# import ProteinMPNN
# from symdesign.third_party import python_codon_tables
# Must add third_party directory to global module namespace before import of DnaChisel
# because of the use of python_codon_tables as a dependency for DnaChisel (not in conda though...)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import DnaChisel
