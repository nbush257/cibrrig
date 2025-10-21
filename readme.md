# CIBRRIG

## :book: [ReadTheDocs](https://cibrrig.readthedocs.io/)
#### Support for extraction, preprocessing, sorting, analysis and plotting of physiology and Neuropixel recordings from rig to fig


<video src=https://github.com/user-attachments/assets/9d79f0d9-8c97-41cb-890a-edcbcbc12d93 width="200" height="500" style="border-radius: 25px" autoplay>
</video>



## Description
Code to integrate hardware and software on the Neuropixel rig in JMB 971 at Seattle Childrens Research Institute, Center for Integrative Brain Research (SCRI-CIBR)
This code is maintained by Nick Bush in the Ramirez Lab and is subject to change.

The rig is designed to monitor breathing and behavior in a head-fixed mouse while recording from neuropixels throughout the brain. Rig is capable of hot-swap between awake and anesthetized preps.

Incorporates both custom code that is specific for the 971 Rig, and more general analyses that are applicable to Neuropixel recordings of respiratory/physiological systems.


**IMPORTANT** 
This code is designed to work in conjunction with hardware in the the [pyExpControl](https://github.com/nbush257/pyExpControl) repository. Most functionality can be used independantly of this hardware, but the most critical piece is the automatically generated log file that is created during recording with this hardware.
The log file is a `.tsv` file with the name `_cibrrig_<run_name>.g<x>.t<x>.tsv`. It has required columns:
`[label, category, start_time, end_time]`, and optional columns that describe parameters of the events (e.g., frequency, duration...). One could create these logfiles manually if desired, or ignore them entirely, but some functionality will fail.

---
## Installation
Create a virtual environment using mamba/conda.
>[!WARNING]
>If on SCRI networks it is critically important to specify the python version here. This circumvents the SSL issue we have been running into. BE SURE YOU HAVE MODIFIED YOUR .condarc file (in `C:/Users/<user>`) appropriately 

```
mamba create -n cibrrig python=3.12
mamba activate cibrrig
```

Then change directory to a place to install cibrrig locally. 
>[!IMPORTANT]
>If you are on NPX 971 room computer, this has already been cloned and you should just install into your new venv.
>```
>cd C:/helpers/cibrrig
>git pull
>pip install -e . 
>```
>(note the period)
>OTHERWISE, clone the repo:
>```
>cd </path/to/somewhere/reasonable/>
>git clone https://github.com/nbush257/cibrrig
>cd cibrrig
>pip install -e .
>```
>(note the period)
> **Once your virtual (mamba/conda) environment has been set up, `git pull` in the cibrrig directory will update `cibrrig` so you do not have to redo the pip install**

>[!WARNING]
>To do manual spike curation, you will need to install `phy` into a seperate conda/mamba environment due to some dependency issues at the moment
> See: https://github.com/cortex-lab/phy



Then, make sure the GPU is working for Kilosort (See [kilosort install instructions](https://github.com/MouseLand/Kilosort) steps 7 and 8):
>Next, if the CPU version of pytorch was installed (will happen on Windows), remove it with `pip uninstall torch`
>Then install the GPU version of pytorch `conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia`

Make sure you are using the GPU by running the kilosort gui:`python -m kilosort` and confirming the PyTorch device is the GPU and not the CPU: ![](gpu_check.png)

Helper packages (Primarily matlab packages) should live in `C:/helpers` on the NPX computer so they are available to all users. Some functionality relies on these packages, but much is being phased out

These include:
 - [Kilosort](https://github.com/MouseLand/Kilosort) (versions 2,3)
 - Chronux http://chronux.org/
 - Breathmetrics https://github.com/zelanolab/breathmetrics
 - SALT ([Kvitsiani et al. 2013](https://www.nature.com/articles/nature12176))

--- 
## :exclamation: Quick start and Data structure 

### From local computer :computer:
> :warning: This performs all processing on the local computer and ties up the resources. This workflow can get backed up if things go sideways.

> If you have recorded a dataset on the NPX computer you can simply open a command prompt and run:
>```
>mamba activate cibrrig
>npx_run_all
>```
> This will open a GUI that prompts you to choose some options and point to where you want thie files saved.

### From sasquatch (HPC) :monkey:
> :warning: Performing the computation on sasquatch keeps the acquisition rig cleaner
>**First**, compress and backup the dataset with:
>```
>mamba activate cibrrig
>backup </local/run/path> <baker/path>
>```
>Example:
>```
>backup D:/Subjects/mickey_mouse \\baker.childrens.sea.kids/archive/ramirez_j/ramirezlab/alf_data_repo/ramirez/Subjects
>```
>**Second**, sing on to a sasquatch login node and run:
>```
>mamba activate iblenv
> pipeline_hpc </baker/path> --no-qc
>```
>N.B. This rsyncs the data to the sasquatch drive, submits SLURM jobs on sasquatch nodes, then moves the data to the ramirezlab alf repository.

>[!NOTE]
> There is incomplete code to run the pipeline via a series of SSH commands (`run_sasquatch.from_NPX`), but is not finished. 




### Details:

Main entry points can be run from anywhere as long as the package has been pip installed
#### :arrow_right:Pipelines (Commands involved in end to end processing)
`npx_run_all` -  Opens a GUI to performs backup, preprocess, and spikesorting\
`backup <local_run_path> <remote_subjects_path>` - Just performs backup\
`pipeline_hpc <run_path>` - Copy from run path to sasquatch tempdir, run pipeline, move to ramirezlab alf repo
#### Modules (Parts of the pipeline that can be run separately if needed)
`npx_preproc <session_path>` - Just performs preprocessing and extraction.\
`ephys_to_alf <run_path>` - Rename the recorded data to alf format\
`spikesort <session_path>` - run spikesorting\
`convert_ks_to_alf <session_path> <sorter>` - convert sorted neural data from kilosort (i.e., phy) to ALF format. <sorter> is the name of the sorting folder. Likely `kilosort4`\
`ephys_qc <session_path>` - Run IBL ephys qc and plots


In practice, it is easiest to simply run `npx_run_all` after recording. Previously run steps will be skipped or appropriately overwritten. Some users have shortcuts to batch scripts that activate the virtual environment and run this.


## Data structure
We save data in a way consistent with the **O**pen **N**europhysiology **E**nvironment ([**ONE**](https://github.com/int-brain-lab/ONE))
For a detailed description of filenames and structure see:[ONE Naming](https://github.com/int-brain-lab/ONE/blob/main/docs/Open_Neurophysiology_Environment_Filename_Convention.pdf) 


Data should be organized with the following structure:
`./<lab>/Subjects/<subject-id>/<yyyy-mm-dd>/<session_number>`
e.g.:
```
alf_data_repo/
├─ ramirez/
│  ├─ Subjects/
│  │  ├─ leonardo/
│  │  │  ├─ 2024-08-01/
│  │  │  │  ├─ 000/**<- SESSION_PATH**
│  │  │  │  ├─ 001/
│  │  │  ├─ 2024-08-02/
│  │  │  │  ├─ 000/
│  │  ├─ donatello/
│  │  │  ├─ 2024-03-05/
│  │  │  │  ├─ 000/
├─ sessions.pqt
├─ datasets.pqt

```

Data should have filenames like: `spikes.times.npy` of the form `<object>.<attribute>.<ext>`

To work with data, you should set up a `one` instance:

```
from one.api import One
one = One.setup(cache_dir=/path/to/alf_data_repo>)
```
>[!CRITICAL]
> Most commands either take a `run` or a `session` as input. There is an important distinction between a `run` and a `session`. 
> - A `run` is in "SpikeGLX  refers to any number of "gates" as recorded by spikeGLX. This folder structure is: `<subject>/<subject>_g0...`
> - A `session` is in ALF/ONE format and refers to a single gate recorded by SpikeGLX, but processed into the format above. 
> Rule of thumb is, if you are working before spikesorting, you are working with `run` format. If you are after spikesorting, it is `session`


---
### For SCRI/Ramirelab users:
The cache_dir lives on the RSS in:
`/helens.childrens.sea.kids/active/ramirez_j/ramirezlab/alf_data_repo`\
which is mounted on sasquatch as:\
`/data/rss/helens/ramirez_j/ramirezlab`\
We mirror all but the raw ephys data to sasquatch work nodes at:
`/data/hps/assoc/private/medullary/data/alf_data_repo`

---

 Now you can structure analysis scripts around the **ONE** structure. Scripts for analysis of data specific to projects should be maintained seperately from this repo. The user is encouraged to use [brainbox](https://github.com/int-brain-lab/ibllib) to manipulate data. 


---
## Hardware

- IMEC Neuropixels
- Sensapex MPM 
- NI based auxiliary recording
- AM systems 1700 Amplifier
- Buxco pressure sensor
- Legacy Ramirez homebrew hardware integrator (for integrating EMGs)
- Valve manifold for gas presentation
    - 100% O2
    - Room air
    - 10% O2 hypoxia  
    - 5%CO2 hypercapnia
    - 100%N2 anoxia
    - Hering breuer closure valve
- Optogenetics - 2 x Cobalt 473nm, 1x Cobalt 635nm lasers. 
- Arduino based experiment control (inspired by Bpod)
- Chameleon Camera(s) - controlled by a teensy camera pulser
- USV mic
- Olfactometer

---
## Software
- **hardware**: Control, CAD, and diagrams of the rig hardware Currently hosted in its own repository. See https://github.com/nbush257/pyExpControl
    - **pyExperimentControl**: Firmware, gui and scripting of arduino control
- **archiving**: Routines for backing up raw data on the SCRI RSS
- **preprocess**: Extract physiological data, experimental events  
- **sorting**: Spikesorting functions and pipelines
- **postprocess**: Compute secondary analyses that rely on spikesorted data 
    -  e.g. optotagging, coherence and respiratory modulation calculations,axon/soma categorization
- **utils**: General utility functions 
- **analysis**: Singlecell and population analyses.
- **plot**: Frequently reused plotting functions, including latent space plotting
- **videos**: Code to make frequently created videos, including evolution of latent, rasters, and auxiliary data over time.
---
## Primary preprocessing and sorting pipeline
This code provides a simple way to process most of the preprocessing steps necesarry to perform after a neuropixel expriment.

Run the pipeline from the cibrrig root with: `python main_pipeline.py`. This will take several hours.

This pipeline runs:
- Backup and compression of raw data
- Conversion of raw data structure to **ONE** 
- Extraction of auxiliary data
    - Synch data
    - Physiology (e.g. breathing)
    - Camera frames times
    - Laser data
- Spike sorting with Kilosort 4 *via* spikeinterface
    - IBL destriping
    - Motion correction (DREDGE, in Spikeinterface)
    - (Optional) Optogenetic artifact removal
    - Spikesorting
    - QC metrics of the spikesorted data
    - UnitRefine assignment of Noise, MUA, SUA
- Conversion of spikesorted data to ALF format
- Concatenation of multiple triggers of auxiliary data and adjusting of time events across streams
