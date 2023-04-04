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
