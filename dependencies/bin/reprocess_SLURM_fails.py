import os
import subprocess


job_number = input('What job number are you looking to re-do commands for?')

find_command = 'find output/. -name "kmeador_refine.cmd%s_*" -exec grep -l "slurmstepd" {} \\; > refine_errors.txt' % job_number
find_errors = subprocess.call(find_command)

with open("refine_errors.txt", "r") as f:
    all_cmds = []
    for error in f.readlines():
        error = error.split('_')[-1].split('.')[0]
        all_cmds.append(error)

with open("cmd_redo.txt", "w") as f:
    for cmd in all_cmds:
        f.write('%s\n' % cmd)
