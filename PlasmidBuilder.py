import os
from SymDesignUtils import pickle_object, unpickle, get_directory_file_paths
import Bio.Seq


# first, import sequence files, by directory name
# next import plasmid vectors by template file
# thrid, query user for backbone and sequence pairs
# fourth find overlap if any using alignment on n-, c-termini
# concatenate sequence to backbone if perfect match otherwise raise DesignError


insert_files = get_directory_file_paths(args.directory, extension='.fasta')
plasmid_backbones = read_fasta_file(backbone_templates)
plasmids = {seq.id: seq.sequence for seq in plasmid_backbones}
pretty_format_table(plasmids.keys())
