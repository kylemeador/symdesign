#!/bin/bash
# $1 is the design stage, $2 is the  passed as the name of the file where all Rosetta Commands for the current design stage are located
# design=design

# JOB=$SLURM_ARRAY_JOB_ID
ID=$SLURM_ARRAY_TASK_ID # if I can access this in python, this bash file is obsolete

commands=`head -n $(($ID * $2)) $3 | tail -${2}`
# script=`head -n $ID $1 | tail -1`
# echo $ID
# echo $script

python $SymDesign/dependencies/python/CommandDistributor.py --stage $1 --success_file ${1}_stage_pose_success.txt --failures_file ${1}_stage_pose_failures.txt --commands $commands
# # stage=${1%.cmd}

# # if [ $(basename "${1%.cmd}") = $design ]
#     # then
#         # mpirun -np 11 bash $script
# # else
# 	bash $script
# # fi
# bash $script
