#!/bin/bash

if [[ "$1" == "h" ]]; then
	help=True
elif [[ "$1" == "help" ]]; then
	help=True
else
	help=False
fi

if [[ $help = True ]]; then
	echo Use this script to run all designs from Nanohedra Output
	echo Include the tag \'-slurm\' as the first positional command if you want to use with a SLURM enabled cluster
	echo Otherwise run will be completed on your computer with python multiprocessing
	exit 0
else
	echo Starting SymDesign
fi


if [[ "$1" = "-slurm" ]]; then 
	slurm=True
else
	slurm=False
fi

directory=`pwd`

if [[ "$slurm" = "False" ]]; then
	# This will process all Poses and run Rosetta Design muliprocessing on a single machine
	python $SymDesign/SymDesign.py -m $2
else
	# This will first process all Poses creating the requisite information
	python $SymDesign/SymDesign.py -cm $2
	# Next initiate commands to set up the appropriate computer environment to process SLURM ARRAYs 
	# This will be done for each step in the design process
	bash $SymDesign/dependencies/bin/ProcessDesignCommands.sh $directory
fi
