import os
import pickle
import subprocess

mem_usage_file = '/gscratch/kmeador/crystal_design/NanohedraEntry65MinMatched6_2nd/mem_usage_analysis_24494851_consensus-cmd.txt'
array_command_pkl = '/gscratch/kmeador/crystal_design/NanohedraEntry65MinMatched6_2nd/consensus.cmd_ArrayToCommand.pkl'
with open(array_command_pkl, 'rb') as f:
    array_to_command_dict = pickle.load(f)


with open(mem_usage_file, 'r') as f:
    command_mem_dict = {}
    for line in f.readlines():
        command_mem_dict[int(line.strip().split()[0].split('_')[1].split('.')[0])] = line.strip().split()[-2][:-1]

# i = 0
# for entry in command_mem_dict:
#     i += 1
#     if i == 6:
#         break
#     print(command_mem_dict[entry])


command_to_mem_dict = {}
for array in array_to_command_dict:
    command_to_mem_dict[array_to_command_dict[array]] = command_mem_dict[array + 1]


dir_to_mem_dict = {}
for command in command_to_mem_dict:
    dir_to_mem_dict[os.path.dirname(command).rstrip('/')[9:]] = command_to_mem_dict[command]


def file_size(file):
    cmd = ['wc', '-c', file, '|', 'awk', '\'{print $1}\'']
    wc_command = subprocess.run(cmd, capture_output=True)

    return wc_command.stdout.split()[0].decode('utf-8')
    # print(subprocess.list2cmdline(cmd))
    # p = subprocess.Popen(cmd, shell=True)
    
    # return p


dir_to_size_dict = {}
for _dir in dir_to_mem_dict:
    process = file_size(os.path.join(_dir, 'clean_asu.pdb'))
    # outs, errs = process.communicate()
    dir_to_size_dict[_dir] = int(process)

dir_to_mem_and_size_dict = {}
for _dir in dir_to_mem_dict:
	dir_to_mem_and_size_dict[_dir] = {'mem': dir_to_mem_dict[_dir], 'file_size': dir_to_size_dict[_dir]}


with open('/gscratch/kmeador/crystal_design/NanohedraEntry65MinMatched6_2nd/dir_to_mem_and_size.pkl', 'wb') as f:
	pickle.dump(dir_to_mem_and_size_dict, f, pickle.HIGHEST_PROTOCOL)
