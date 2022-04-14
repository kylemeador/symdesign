#!/bin/bash

echo ".
# To start, copy the files in this directory 'scp unit_testing new/path/'. You will need to set up a directory on /yeates1. I suggest this storage device because it is accessible to both escher and cassini.

# Here are some example commands from Nanohedra. You can try them in the various Nanohedra environments by changing the program location, i.e. /yeates1/kmeador/symdesign/Nanohedra.py to /yeates1/kmeador/nanohedra/Nanohedra.py. If you change the program source, you will have to modify the -pdb_dir[1/2]_path argument to be \'-pdb_dir1_path C3/\' instead of C3/3tcr.pdb1. The original version couldn't take individual files, instead it took directories. This is one change that I have made to make the program opperate in parallel.

# I have written below all the commands assuming you have a directory called \'kaylie_nanohedra_testing\' in your /yeates1/kbair directory. Ensure this is correct or change the commands

# Another point. The environment for each version of the program is slightly different. I advise you to set up miniconda on escher (eventually cassini) so that you can manage this. To run with the /symdesign/Nanohedra.py version, you will need to execute the command \'conda env create --file /yeates1/kmeador/symdesign/SymDesignEnvironment.yaml\'. You can do the same thing for the /nanohedra/Nanohedra.py version with the --file \'/yeates1/kmeador/nanohedra/NanohedraEnvironment.yaml\'

# T33 - Point group
python /yeates1/kmeador/symdesign2/Nanohedra.py -dock -entry 54 -pdb_dir1_path C3/3tcr.pdb1 -pdb_dir2_path C3/3mjz.pdb1 -rot_step1 3 -rot_step2 3 -outdir /yeates1/kmeador/symdesign2/unit_testing/NanohedraEntry54DockedPoses -output_assembly

# p222 - Layer group (no rotational degrees of freedom)
python /yeates1/kmeador/symdesign2/Nanohedra.py -dock -entry 84 -pdb_dir1_path D2/3wb0.pdb1 -pdb_dir2_path D2/1rcu.pdb1 -outdir /yeates1/kmeador/symdesign2/unit_testing/NanohedraEntry84DockedPoses -output_assembly

# P432 - Space group (pdb1 rotational degree of freedom)
python /yeates1/kmeador/symdesign2/Nanohedra.py -dock -entry 67 -pdb_dir1_path C3/1vhc.pdb1 -pdb_dir2_path D4/2fw7.pdb1 -rot_step1 3 -outdir /yeates1/kmeador/symdesign2/unit_testing/NanohedraEntry67DockedPoses -output_assembly

# To run these actual scripts, comment out the echo and \'\" \"\' lines.
"
