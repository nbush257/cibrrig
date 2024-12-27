# CIBRRIG
**Updated 2024-09-07**

[ReadTheDocs](https://cibrrig.readthedocs.io/)

#TODO: Currently the log file does not get copied in the archiving. Need to:

- [ ] Archive the log file in all future recordings
- [ ] Archive the log file in all previously recorded data.

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
>If on SCRI networks it is critically important to specify the python version here. This circumvents the SSL issue we have been running into

`mamba create -n cibrrig python=3.12`

Then change directory to a place to install cibrrig locally. 
>[!IMPORTANT]
>If you are on NPX 971 room computer, this has already been cloned and you should just install into your new venv.
>```
>cd C:/helpers/cibrrig
>git pull
>pip install -e .
>```
>OTHERWISE, clone the repo:
>```
>cd </path/to/somewhere/reasonable/>
>git clone https://github.com/nbush257/cibrrig
>cd cibrrig
>pip install -e .
>```
> **Once your virtual (mamba/conda) environment has been set up, `git pull` in the cibrrig directory will update `cibrrig` so you do not have to redo the pip install**


Helper packages (Primarily matlab packages) should live in `C:/helpers` on the NPX computer so they are available to all users. Some functionality relies on these packages.

These include:
 - [Kilosort](https://github.com/MouseLand/Kilosort) (versions 2,3)
 - Chronux http://chronux.org/
 - Breathmetrics https://github.com/zelanolab/breathmetrics
 - SALT ([Kvitsiani et al. 2013](https://www.nature.com/articles/nature12176))

--- 
## Quick start and Data structure 

**Quickstart** - From recording to data in two lines
```
mamba activate cibrrig
npx_run_all
```

### Details:

Main entry points can be run from anywhere as long as the package has been pip installed\
`npx_run_all` -  Opens a GUI to performs backup, preprocess, and spikesorting\
`backup` - Just performs backup\
`npx_preproc <session_path>` - Just performs preprocessing and extraction.\
`ephys_to_alf <run_path>` - Rename the recorded data to alf format\
`spikesort <session_path>` - run spikesorting\
`convert_ks_to_alf <session_path> <sorter>` - convert sorted neural data from kilosort (i.e., phy) to ALF format. <sorter> is the name of the sorting folder. Likely `kilosort4`

In practice, it is easiest to simply run `npx_run_all` after recording. Previously run steps will be skipped or appropriately overwritten. Some users have shortcuts to batch scripts that activate the virtual environment and run this.

> [!NOTE]  
This sorts the data, but does not convert the sorted data to `alf` format in case the user needs to do a manual curation in  `phy` first. Once the manual curation is done, the user needs to run: `postprocess.convert_ks_to_alf` on the session. This can be done easily with:
>```
>cd </path/to/session>
>convert_ks_to_alf ./ <sorter>
>```
>where `<sorter>` is the name of the sorting folder. Should be `kilosort4`

**Data structure**\
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
- Conversion of data structure to **ONE** 
- Extraction of auxiliary data
    - Synch data
    - Physiology (e.g. breathing)
    - Camera frames times
    - Laser data
- Spike sorting with Kilosort 4 *via* spikeinterface
    - IBL destriping
    - Motion estimation (for plotting, in Spikeinterface)
    - Motion correction (in KS4)
    - (Optional) Optogenetic artifact removal
    - QC metrics of the spikesorted data
- Concatenation of multiple triggers of auxiliary data

Many of the above pipeline elements can be run independently by the code in `./preprocess/`

---

##### At this point, any manual curation of the spike sorting can be done in phy. Steps after this will "freeze" the spike sorting, so any changes to cluster assignment will require a recomputation
