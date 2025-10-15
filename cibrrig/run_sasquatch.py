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
import click

# === SET UP PATHS ===
SASQUATCH_WORKING_DIR = Path("/data/hps/assoc/private/medullary/data/sorting_dir")
BAKER_DEST = Path(
    r"\\baker.childrens.sea.kids\archive\ramirez_j\ramirezlab\alf_data_repo\ramirez\Subjects"
)
BAKER_DEST_RSS = Path(
    "/data/rss/baker/ramirez_j/ramirezlab/alf_data_repo/ramirez/Subjects"
)
HELENS_DEST = Path(
    "/data/rss/helens/ramirez_j/ramirezlab/alf_data_repo/ramirez/Subjects"
)


# As is, not the most resource efficient. Would be better to to  request fewer cpu when on GPU, vice versa.
job_params = {
    "nodes": 1,
    "ntasks_per_node": 1,
    "cpus_per_task": 64,
    "gpus": 1,
    "partition": "gpu-core-sponsored",
    "memory": "128G",
    "walltime": "24:00:00",
    "association": "gpu-medullary-sponsored",
}


def gen_sbatch_script(
    job_params: dict,
    run_folder: Path,
    helens_dest: Path | None = None,
    baker_dest_rss: Path | None = None,
    opto_flag=True,
    QC_flag=True,
) -> str:
    """Generate SLURM batch script as a string.

    Args:
        job_params (dict): Dictionary containing job parameters.
        run_folder (Path): Path to the run folder on the remote server.
        helens_dest (Path): Path to the destination folder on Helen's server.
        baker_dest_rss (Path): Path to the destination folder on Baker's RSS server.

    Returns:
        str: SLURM batch script as a string.
    """

    baker_dest_rss = "" if baker_dest_rss is None else baker_dest_rss.as_posix()
    helens_dest = "" if helens_dest is None else helens_dest.as_posix()
    runname = "npx_pipeline_" + run_folder.name
    run_folder = run_folder.as_posix()
    opto_flag = "-O" if opto_flag else ""
    QC_flag = "-Q" if QC_flag else ""

    sbatch_script = f"""#!/bin/bash
#SBATCH --nodes={job_params["nodes"]}
#SBATCH --ntasks-per-node={job_params["ntasks_per_node"]}
#SBATCH --cpus-per-task={job_params["cpus_per_task"]}
#SBATCH --gpus={job_params["gpus"]}
#SBATCH --mem={job_params["memory"]}
#SBATCH --time={job_params["walltime"]}
#SBATCH --account={job_params["association"]}
#SBATCH --partition={job_params["partition"]}
#SBATCH --job-name={runname}

source activate iblenv
npx_run_all_no_gui {run_folder} {helens_dest} {baker_dest_rss} {opto_flag} {QC_flag} 
    """
    return sbatch_script

def gen_tarball_script(run_folder: Path) -> str:
    """Generate SLURM batch script as a string.

    Args:
        run_folder (Path): Path to the run folder on the remote server.
    Returns:
        str: SLURM batch script as a string.
    """
    runname = "tarball_spikeinterface_" + run_folder.name
    run_folder = run_folder.as_posix()

    sbatch_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G   
#SBATCH --time=12:00:00
#SBATCH --account=cpu-medullary-sponsored
#SBATCH --partition=cpu-core-sponsored
#SBATCH --job-name={runname}_tarball
source activate iblenv
tarball_spikeinterface {run_folder}
    """
    return sbatch_script

def remove_empty_folders(root_dir):
    print(f"Removing empty folders in {root_dir.as_posix()}")
    for path, dirs, files in os.walk(root_dir, topdown=False):
        # Iterate from the deepest directories first (topdown=False)
        for name in dirs:
            dir_path = os.path.join(path, name)
            if not os.listdir(dir_path):  # Check if the directory is empty
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
                except OSError as e:
                    print(f"Error removing {dir_path}: {e}")
    os.rmdir(root_dir)

@click.group()
def main():
    pass

@main.command()
@click.argument("local_run_path", type=click.Path(exists=True))
def from_NPX(
    local_run_path,
    sasquatch_working_dir=SASQUATCH_WORKING_DIR,
    helens_dest=HELENS_DEST,
    baker_dest_rss=BAKER_DEST_RSS,
    opto_flag=True,
    QC_flag=True,
):
    """ Run the pipeline on sasquatch from the acquisition computer using SSH commands

    Args:
        local_run_path (Path): Path to the local run directory.
        sasquatch_working_dir (Path, optional): Path to sasquatch working directory. Defaults to SASQUATCH_WORKING_DIR.
        helens_dest (Path, optional): Path to Helen's destination directory. Defaults to HELENS_DEST.
        baker_dest_rss (Path, optional): Path to Baker's RSS destination directory. Defaults to BAKER_DEST_RSS.
        opto_flag (bool, optional): Whether to remove opto artifact. Defaults to True.
        QC_flag (bool, optional): Whether to run ephys QC. Defaults to True.

    Returns:
        None
        
    Raises:
        FileNotFoundError: If the local run path does not exist.
    """
    user = os.environ.get("USERNAME", os.environ.get("USER", None))
    run_folder = SASQUATCH_WORKING_DIR.joinpath(local_run_path.name)
    # ==========
    # Backup
    backup.no_gui(local_run_path, BAKER_DEST, compress_locally=True)
    print("\n")
    print("=" * 20)
    print("Backup complete Copying to sasquatch.")

    # ==========
    # Copy to sasquatch
    process = subprocess.Popen(
        f"scp -r {local_run_path} {user}@login-1.hpc.childrens.sea.kids:{SASQUATCH_WORKING_DIR.as_posix()} ",
        shell=True,
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error: SCP command failed with return code {process.returncode}")
        print(stdout.decode())
        print(stderr.decode())
        exit(process.returncode)

    print("\n")
    print("=" * 20)
    print("Data copy complete")

    # Write batch script locally
    batch_script = gen_sbatch_script(
        job_params, run_folder, HELENS_DEST, BAKER_DEST_RSS
    )
    with open("run_job.sh", "w", newline="\n") as f:
        f.write(batch_script)

    # Copy batch script to remote run folder
    process = subprocess.Popen(
        f"scp run_job.sh {user}@login-1.hpc.childrens.sea.kids:{run_folder.as_posix()} ",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    print("\n")
    print("=" * 20)
    print("Copy sbatch file complete. Submitting SLURM job.")

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

@main.command()
@click.argument("run_src", type=click.Path(exists=True))
@click.option("--sasquatch_working_dir", type=click.Path(), default=SASQUATCH_WORKING_DIR, help="Path to sasquatch working directory where sorting will occur.")
@click.option("--helens_dest", type=click.Path(), default=HELENS_DEST, help="Path to Helens directory where data will be moved after sorting.")
@click.option("--opto/--no-opto", default=True, help="Whether to remove opto artifact.")
@click.option("--qc/--no-qc", default=True, help="Whether to run ephys QC.")
def from_baker(
    run_src,
    sasquatch_working_dir=SASQUATCH_WORKING_DIR,
    helens_dest=HELENS_DEST,
    opto=True,
    qc=True,
):
    '''
    Run the pipeline on sasquatch from data already on baker.
    Must be run from the login node.

    This script is accessed by the entry point `pipeline_hpc` after installing cibrrig in a conda environment.
    e.g.:
        pipeline_hpc /data/rss/baker/ramirez_j/ramirezlab/alf_data_repo/ramirez/Subjects/m2025-01

    Best practice is to create a tmux session on login node since you will move large files, activate iblenv, then run this 

    Args:
        run_src (str): Source run path, e.g. '/data/rss/baker/ramirez_j/ramirezlab/alf_data_repo/ramirez/Subjects/m2025-01'
        sasquatch_working_dir (Path, optional): Path to sasquatch working directory. Defaults to SASQUATCH_WORKING_DIR.
        helens_dest (Path, optional): Path to Helen's destination directory. Defaults to HELENS_DEST.
        opto_flag (bool, optional): Whether to remove opto artifact. Defaults to True.
        QC_flag (bool, optional): Whether to run ephys QC. Defaults to True.
    Raises:
        FileNotFoundError: If the baker path does not exist.
    

    '''
    run_src = Path(run_src)
    if not run_src.exists():
        raise FileNotFoundError(f"Run path {run_src} does not exist.")
    # RSYNC FROM SRC TO SASQUATCH
    temp_dir = sasquatch_working_dir.joinpath(run_src.name)
    rsync_cmd_src_to_sasquatch = (
        f"rsync -rzP {run_src.as_posix()}/ {temp_dir.as_posix()}/"
    )
    subprocess.run(rsync_cmd_src_to_sasquatch, shell=True, check=True)

    # ============== #
    # Generate batch script
    print('Generating batch script')
    batch_script = gen_sbatch_script(
        job_params,
        temp_dir,
        opto_flag=opto,
        QC_flag=qc,
    )
    batch_fn = temp_dir.joinpath("run_job.sh")
    with open(batch_fn, "w", newline="\n") as f:
        f.write(batch_script)

    # ============== #
    # Submit job with sbatch
    slurm_submit_cmd = f"cd {temp_dir.as_posix()} && sbatch --wait run_job.sh"
    result = subprocess.run(slurm_submit_cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: SLURM job submission failed with return code {result.returncode}")
        exit(result.returncode)

    # ============== #
    # Perform tarballing on sasquatch
    tarball_script = gen_tarball_script(temp_dir)
    tarball_fn = temp_dir.joinpath("tarball_job.sh")
    with open(tarball_fn, "w", newline="\n") as f:
        f.write(tarball_script)
    tarball_submit_cmd = f"cd {temp_dir.as_posix()} && sbatch --wait tarball_job.sh"
    result = subprocess.run(tarball_submit_cmd, shell=True, check=True)
    if result.returncode != 0:
        print(f"Error: Tarball job submission failed with return code {result.returncode}")
        exit(result.returncode)
    
    # ============== #
    # rsync from sasquatch to helens
    rsync_cmd_sasquatch_to_helens = (
        f"rsync -rzP --remove-source-files {temp_dir.as_posix()} {helens_dest.as_posix()}"
    )

    subprocess.run(rsync_cmd_sasquatch_to_helens, shell=True, check=True)
    
    # Remove temp dir on sasquatch
    remove_empty_folders(temp_dir)


if __name__ == "__main__":
    main()