#!/bin/bash
# $1 is passed as the name of the file where all Rosetta Commands for the current design stage are located
design=design

JOB=$SLURM_ARRAY_JOB_ID
ID=$SLURM_ARRAY_TASK_ID

script=`head -n $ID $1 | tail -1`
echo $ID
echo $script
# stage=${1%.cmd}

# if [ $(basename "${1%.cmd}") = $design ]
    # then
        # mpirun -np 11 bash $script
# else
	bash $script
# fi
