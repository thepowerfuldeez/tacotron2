import time
import torch
import sys
import subprocess

argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
# argslist.append('--n_gpus={}'.format(num_gpus))
workers = []
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--group-name=group_{}".format(job_id))

for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))
    stdout = None if i == 0 else open(f"logs/{job_id}_GPU_{i}.log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    p.wait()