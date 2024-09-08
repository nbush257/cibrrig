# CIBRRIG
**Updated 2024-09-07**

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
(Recommended): Use a virtual environment like conda

Change directory to the path with the `setup.py` file
`cd /path/to/cibrrig`
Install using pip
`pip install -e .`

This should install the `iblenv` dependencies as well. 

Helper packages (Primarily matlab packages) should live in `C:/helpers` on the NPX computer so they are available to all users. Some functionality relies on these packages.

These include:
 - Kilosort (versions 2,3)
 - Chronux http://chronux.org/
 - Breathmetrics https://github.com/zelanolab/breathmetrics
 - SALT ([Kvitsiani et al. 2013](https://www.nature.com/articles/nature12176))

--- 
## Quick start and Data structure 

**Quickstart** - From recording to data

First activate the virtual environment you installed cibrrig on (e.g. `mamba activate iblenv`)

Main entry points can be run from anywhere as long as the package has been pip installed\
`npx_run_all` -  Opens a GUI to performs backup, preprocess, and spikesorting\
`backup` - Just performs backup\
`npx_preproc <session_path>` - Just performs preprocessing and extraction.\
`ephys_to_alf <run_path>` - Rename the recorded data to alf format
`spikesort <session_path>` - run spikesorting

In practice, it is easiest to simply run `npx_run_all` after recording. Previously run steps will be skipped or appropriately overwritten. Some users have shortcuts to batch scripts that activate the virtual environment and run this.

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
- **hardware**: Control, CAD, and diagrams of the rig hardware
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
