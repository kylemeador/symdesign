#!/bin/bash
# $1 is passed as the name of the file where all Rosetta Commands for the current design stage are located
#SBATCH --job-name=symdesign_sbatch_test
#SBATCH --output=%directory_%a.out
#SBATCH --error=%directory_%a.er
#SBATCH --partition=long
#SBATCH --nodes=10
##SBATCH --ntasks=60
##SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kmeador
#SBATCH --array=20-145

`bash diSbatch.sh $1`
