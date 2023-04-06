from __future__ import annotations

import os
import subprocess

import jax


def check_gpu() -> list[str] | bool:
    available_devices = jax.local_devices()
    devices = []
    for idx, device in enumerate(available_devices):
        if device.platform == 'gpu':
            # self.gpu_available = True  # Todo could be useful
            devices.append(device.device_kind)
            # device_id = idx
            # return True
    if devices:
        return devices
    else:
        return False

# 2-1,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-1,11G,11G
# 2-2,NVIDIA GeForce RTX 3090
# 2-2,24G
# 2-3,NVIDIA GeForce GTX 1080 Ti
# 2-3,11G
# 2-4,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-4,11G,11G
# 2-5,NVIDIA GeForce GTX 1080 Ti
# 2-5,11G
# 2-6,?
# 2-7,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-7,11G
# 2-8,NVIDIA TITAN RTX
# 2-8,24G
# 2-9,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-9,11G,11G
# 2-10,NVIDIA GeForce RTX 3080
# 2-10,10G
# 2-21,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-21,11G,11G
# 2-22,NVIDIA GeForce GTX 1080,NVIDIA GeForce GTX 1080
# 2-22,8G,8G
# 2-23,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-23,11G,11G
# 2-24,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-24,11G,11G
# 2-25,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-25,11G,11G
# 2-26,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-26,11G,11G
# 2-27,NVIDIA GeForce GTX 1080,NVIDIA GeForce GTX 1080
# 2-27,8g,8G
# 2-28,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-26,11G,11G
# 2-29,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-29,11G,11G
# 2-30,NVIDIA GeForce GTX 1080 Ti,NVIDIA GeForce GTX 1080 Ti
# 2-30,11G,11G
# 2-31,NVIDIA GeForce GTX 1080 Ti
# 2-31,11G
# 2-32,NVIDIA GeForce RTX 2080 Ti
# 2-32,11G
# 2-33,NVIDIA GeForce RTX 2080 Ti
# 2-33,11G
# 2-34,NVIDIA GeForce RTX 2080 Ti
# 2-34,11G
# 2-35,NVIDIA GeForce RTX 2080 Ti
# 2-35,11G
# 2-36,NVIDIA GeForce RTX 2080 Ti
# 2-36,11G
# 2-37,NVIDIA GeForce RTX 2080 Ti
# 2-37,11G
# 2-38,NVIDIA GeForce RTX 2080 Ti
# 2-38,11G
# 2-39,NVIDIA GeForce RTX 2080 Ti
# 2-39,11G
# 2-40,NVIDIA GeForce RTX 2080 Ti
# 2-40,11G
# 2-41,NVIDIA GeForce RTX 2080 Ti
# 2-41,11G
# 2-42,NVIDIA GeForce RTX 2080 Ti
# 2-42,11G
# 2-43,NVIDIA GeForce RTX 2080 Ti
# 2-43,11G
# 2-44,NVIDIA GeForce RTX 2080 Ti
# 2-44,11G
# 2-45,NVIDIA GeForce RTX 2080 Ti
# 2-45,11G
# 2-46,NVIDIA GeForce RTX 2080 Ti
# 2-46,11G
# 2-47,NVIDIA GeForce RTX 2080 Ti
# 2-47,11G
# 2-48,NVIDIA GeForce RTX 2080 Ti
# 2-48,11G
# 6-16,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti
# 6-16,12G,12G,12G,12G
# 6-17,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti
# 6-17,12G,12G,12G,12G
# 6-18,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti
# 6-18,12G,12G,12G,12G
# 6-19,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti
# 6-19,12G,12G,12G,12G
# 6-20,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti,NVIDIA GeForce RTX 3080 Ti
# 6-20,12G,12G,12G,12G
# 6-21,NVIDIA GeForce RTX 3090
# 6-21,24G
# 6-22,NVIDIA Graphics Device
# 6-22,24G
# 6-23,??


if __name__ == '__main__':
    gpu_type = check_gpu()
    if gpu_type:
        if 'SLURM_JOB_ID' in os.environ:
            jobid = os.environ['SLURM_JOB_ID']  # SLURM_JOB_ID
            # array_jobid = os.environ.get('SLURM_ARRAY_TASK_ID')
            # if array_jobid:
            #     jobid = f'{jobid}_{array_jobid}'  # SLURM_ARRAY_TASK_ID
            if 'SLURM_ARRAY_TASK_ID' in os.environ:
                jobid = f'{jobid}_{os.environ["SLURM_ARRAY_TASK_ID"]}'  # SLURM_ARRAY_TASK_ID
            #     logger.debug(f'The job is managed by SLURM with SLURM_ARRAY_TASK_ID={jobid}')
            # else:
            #     logger.debug(f'The job is managed by SLURM with SLURM_JOB_ID={jobid}')
            # Run the command 'scontrol show job {jobid}'
            p = subprocess.Popen(['scontrol', 'show', 'job', jobid], stdout=subprocess.PIPE)
            out, err = p.communicate()
            out = out.decode('UTF-8')
            """Searching for the line
            Partition=gpu AllocNode:Sid=cassini:21698
            ReqNodeList=(null) ExcNodeList=compute-2-[3,5-7,10,22,27,31-48]
            *NodeList=compute-2-28
            BatchHost=compute-2-28
            NumNodes=1 NumCPUs=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
            """
            start_index = out.find('NodeList=') + 9  # <- 9 is length of search string
            # start_index = out.find('Nodes=') + 6  # <- 6 is length of search string
            node_allocated = out[start_index:start_index + 15].split()[0]
        else:
            # raise RuntimeError(f'Not running in SLURM environment')
            node_allocated = ''

        print(f'{node_allocated},{",".join(gpu_type)}')
