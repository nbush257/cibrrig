"""
Need:
1) Local run path
2) Sasquatch directory
3) active ALF target
4) Archive ALF target

Steps:
1) Compress local
2) Backup to Baker
3) Copy to Sasquatch
4) Remove local
5) Submit slurm job
5) Move to helens
"""

from pathlib import Path
import subprocess
import os
from cibrrig.archiving import backup
import time

# === SET UP PATHS ===
LOCAL_RUN_PATH = Path(r"D:\Subjects\test_subject")
SASQUATCH_WORKING_DIR = Path("/data/hps/assoc/private/medullary/data/sorting_dir")
BAKER_DEST = Path(
    r"\\baker.childrens.sea.kids\archive\ramirez_j\ramirezlab\alf_data_repo\ramirez\Subjects"
)
BAKER_DEST_RSS = Path('/data/rss/baker/ramirez_j/ramirezlab/alf_data_repo/ramirez/Subjects')
HELENS_DEST = Path("/data/rss/helens/ramirez_j/ramirezlab/alf_data_repo/ramirez/Subjects")

assert LOCAL_RUN_PATH.exists()

# As is, not the most resource efficient. Would be better to to  request fewer cpu when on GPU, vice versa.
# === JOB PARAMS ===
ASSOC = " gpu-medullary-sponsored"
user = os.environ["USERNAME"]

run_name = LOCAL_RUN_PATH.name
run_folder = SASQUATCH_WORKING_DIR.joinpath(run_name)
nodes = 1
ntasks_per_node = 1
cpus_per_task = 64
gpus = 1
partition = "gpu-core-sponsored"
memory = "128G"
walltime = "08:00:00"


# ==========
# Backup
backup.no_gui(LOCAL_RUN_PATH, BAKER_DEST, compress_locally=True)
print('\n')
print('='*20)
print('Backup complete Copying to sasquatch.')

# ==========
# Copy to sasquatch
process = subprocess.Popen(
    f"scp -r {LOCAL_RUN_PATH} {user}@login-1.hpc.childrens.sea.kids:{SASQUATCH_WORKING_DIR.as_posix()} ",
    shell=True,
)
stdout, stderr = process.communicate()

if process.returncode != 0:
    print(f"Error: SCP command failed with return code {process.returncode}")
    print(stdout.decode())
    print(stderr.decode())
    exit(process.returncode)

print('\n')
print('='*20)
print('Data copy complete')


batch_script = f"""#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus={gpus}
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH -A {ASSOC}
#SBATCH --partition={partition}

source activate iblenv
npx_run_all_no_gui {run_folder.as_posix()} {HELENS_DEST.as_posix()} {BAKER_DEST_RSS.as_posix()}

"""
# Write batch script locally 
with open("run_job.sh", "w", newline="\n") as f:
    f.write(batch_script)
time.sleep(0.1)

# Copy batch script to remote run folder
process = subprocess.Popen(
    f"scp run_job.sh {user}@login-1.hpc.childrens.sea.kids:{run_folder.as_posix()} ",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
).communicate()
print('\n')
print('='*20)
print('Copy sbatch file complete. Submitting SLURM job.')


# Submit job with sbatch
ssh_cmd = f'ssh {user}@login-1.hpc.childrens.sea.kids "cd \\"{run_folder.as_posix()}\\" && sbatch run_job.sh"'
process = subprocess.Popen(
    ssh_cmd,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

stdout, stderr = process.communicate()
if process.returncode != 0:
    print(f"Error: SSH command failed with return code {process.returncode}")
    print(stdout.decode())
    print(stderr.decode())
    exit(process.returncode)
else:
    print(stdout.decode())
    print(stderr.decode())
    print("SLURM job submitted successfully.")
os.remove("run_job.sh")
    
