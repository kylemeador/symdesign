#!/bin/bash
# $1 is passed as the name of the directory where all Rosetta Commands for design are located
sbatch=_sbatch.sh
stages=(refine design filter)  # metrics
scripts=(run_${stages[0]}${sbatch} run_${stages[1]}${sbatch} run_${stages[2]}${sbatch}) # run_${stages[3]}${sbatch})
python $SymDesign/WriteSBATCH.py -f ${1}/${stages[0]}.cmd ${scripts[0]} --default
python $SymDesign/WriteSBATCH.py -f ${1}/${stages[1]}.cmd ${scripts[1]} --default  # --tasks 11 --mem 4000
python $SymDesign/WriteSBATCH.py -f ${1}/${stages[2]}.cmd ${scripts[2]} --default
# python $SymDesign/WriteSBATCH.py -f ${1}/${stages[3]}.cmd ${scripts[3]} --default

# Next, make a script which executes each of these sbatch scripts
#design+="#!/bin/bash"
echo "#!/bin/bash" > ${1}/SymDesign_Design.sh
for script in ${scripts[@]}
do
    echo "sbatch $script" >> ${1}/SymDesign_Design.sh
done

# Finally, run the script!
bash ${1}/SymDesign_Design.sh
