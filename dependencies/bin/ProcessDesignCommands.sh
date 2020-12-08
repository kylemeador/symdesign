#!/bin/bash
# $1 is passed as the name of the directory where all Rosetta Commands for design are located, $2 provides MPI support

if [ $2 == 'mpi' ]; then
	sbatch=_${2}_sbatch.sh
	3=1 --cpus-per-task $(($3 / 2)) # $3
else
	sbatch=_sbatch.sh
	3=1
fi

stages=(refine design metrics analysis)
# scripts=(run_${stages[0]}${sbatch} run_${stages[1]}${sbatch} run_${stages[2]}${sbatch} run_${stages[3]}${sbatch})
# python $SymDesign/dependencies/python/WriteSBATCH.py -f ${1}/${stages[0]}.cmd ${scripts[0]} --default --threads-per-core 2 --mem_per_cpu 8000
# python $SymDesign/dependencies/python/WriteSBATCH.py -f ${1}/${stages[1]}.cmd ${scripts[1]} --default --ntasks $3 --threads-per-core 2 --mem_per_cpu 4000 # 
# python $SymDesign/dependencies/python/WriteSBATCH.py -f ${1}/${stages[2]}.cmd ${scripts[2]} --default --ntasks $3 --threads-per-core 2 --mem_per_cpu 1000
# python $SymDesign/dependencies/python/WriteSBATCH.py -f ${1}/${stages[2]}.cmd ${scripts[3]} --default --threads-per-core 2  # can I make python use multiple cores for one process?

# or we could just use bash...
mkdir ${1}/output
number_cmds=`wc -l ${1}/${stages[0]}.cmd | awk '{print $1}'`
echo Total, $number_cmds commands will be processed
# refine
#`sbatch --job-name=${USER}_${stages[0]}.cmd --output=${1}/output/%A_%a --error=${1}/output/%A_%a --partition=long --ntasks=1 --threads-per-core 2 --mem_per_cpu 8000 --array=1-${number_cmds}%800 --no-requeue bash $SymDesign/dependencies/bin/diSbatch.sh ${1}/${stages[0]}.cmd`
echo sbatch --job-name=${USER}_${stages[0]}.cmd --output=${1}/output/%A_%a --error=${1}/output/%A_%a --partition=long --ntasks=1 --threads-per-core 2 --mem_per_cpu 8000 --array=1-${number_cmds}%800 --no-requeue bash $SymDesign/dependencies/bin/diSbatch.sh ${1}/${stages[0]}.cmd
# design
#`sbatch --job-name=${USER}_${stages[1]}.cmd --output=${1}/output/%A_%a --error=${1}/output/%A_%a --partition=long --ntasks=$3 --threads-per-core 2 --mem_per_cpu 4000 --array=1-${number_cmds}%800 --no-requeue bash $SymDesign/dependencies/bin/diSbatch.sh ${1}/${stages[1]}.cmd`
echo sbatch --job-name=${USER}_${stages[1]}.cmd --output=${1}/output/%A_%a --error=${1}/output/%A_%a --partition=long --ntasks=$3 --threads-per-core 2 --mem_per_cpu 4000 --array=1-${number_cmds}%800 --no-requeue bash $SymDesign/dependencies/bin/diSbatch.sh ${1}/${stages[1]}.cmd
# metrics
#`sbatch --job-name=${USER}_${stages[2]}.cmd --output=${1}/output/%A_%a --error=${1}/output/%A_%a --partition=long --ntasks=$3 --threads-per-core 2 --mem_per_cpu 1000 --array=1-${number_cmds}%800 --no-requeue bash $SymDesign/dependencies/bin/diSbatch.sh ${1}/${stages[2]}.cmd`
echo sbatch --job-name=${USER}_${stages[2]}.cmd --output=${1}/output/%A_%a --error=${1}/output/%A_%a --partition=long --ntasks=$3 --threads-per-core 2 --mem_per_cpu 1000 --array=1-${number_cmds}%800 --no-requeue bash $SymDesign/dependencies/bin/diSbatch.sh ${1}/${stages[2]}.cmd


# # Next, make a script which executes each of these sbatch scripts
# #design+="#!/bin/bash"
# echo "#!/bin/bash" > ${1}/SymDesign_Design.sh
# for script in ${scripts[@]}
# do
#     echo "sbatch $script" >> ${1}/SymDesign_Design.sh
# done

# # Finally, run the script!
# bash ${1}/SymDesign_Design.sh
