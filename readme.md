# CIBRRIG
---
**Updated 2024-04-08**
## Description
Code to integrate the hardware and software on the Neuropixel rig in JMB 971 at Seattle Childrens Research Institute, Center for Integrative Brain Research (SCRI-CIBR)
This code is maintained by Nick Bush in the Ramirez Lab and is subject to change.

The rig is designed to monitor breathing and behavior in a head-fixed mouse while recording from neuropixels throughout the brain. Rig is capable of hot-swap between awake and anesthetized preps.

Code *should* be executable both locally or on the HPC (cybertron, sasquatch coming soon)

---
## Installation
(Recommended): Use a virtual environment like conda
Change directory to the path with the `setup.py` file
`cd /path/to/cibrrig`
Install using pip
`pip install .`

This should install the `iblenv` dependencies as well. 

Helper packages (Packages from other groups (e.g., kilosort)) should live in `C:/helpers` on the NPX computer so they are available to all users


--- 
## Data structure and quickstart

**WARNING** These are currently broken
Main entry points can be run from anywhere as long as the package has been pip installed
`npx_run_all` -  Performs backup, preprocess, and spikesorting
`backup` - Just performs backup
`npx_preproc <session_path>` - Just performs preprocessing and extraction.


We will save data in a way consistent with the **O**pen **N**europhysiology **E**nvironment ([**ONE**](https://github.com/int-brain-lab/ONE))
For a detailed description of filenames and structure see:[ONE Naming](https://github.com/int-brain-lab/ONE/blob/main/docs/Open_Neurophysiology_Environment_Filename_Convention.pdf) 


Data should be organized with the following structure:
`./<project>/<lab>/Subjects/<subject-id>/<yyyy-mm-dd>/<session_number>`
e.g.:
```
project/
├─ data/
│  ├─ mouse001/
│  │  ├─ 2024-02-01/
│  │  │  ├─ 000/
│  │  │  ├─ 001/
│  │  ├─ 2024-04-01/
│  │  │  ├─ 000/
│  ├─ mouse002/
│  │  ├─ 2024-03-12/
│  │  │  ├─ 000/
```

Data should have filenames like: `spikes.times.npy` of the form `<object>.<attribute>.<ext>`

To work with data, you should set up a `one` instance:

```
from one.api import One
one = One.setup(cache_dir=/path/to/<project>)
```
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

*In progress*: 
- Olfactometer

*TODO*:

---
## Software
- **archiving**: Routines for backing up raw data on the SCRI RSS
- **preprocess**: Extract physiological data, experimental events  
- **sorting**: Spikesorting functions and pipelines
- **hardware**: Control, CAD, and diagrams of the rig hardware
    - **pyExperimentControl**: Firmware, gui and scripting of arduino control
- **postprocess**: Compute secondary analyses that rely on spikesorted data 
    -  e.g. optotagging, coherence calculations,axon/soma categorization

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

---
## Secondary postprocessing
These steps involve operations on spiking data. They should without exception run on a "session_path" which is always of the form:
```
project/
├─ data/
│  ├─ mouse001/
│  │  ├─ 2024-02-01/
│  │  │  ├─ 000/ **<- SESSION_PATH**
```
- Concatenate auxiliary data over multiple triggers (Optional - usually done in the preprocessing pipeline)
    - `python -m cibrrig.postprocess.concatenate_triggers <session_path>`
- Compute respiratory coherence (Defaults to computing on the first 5 minutes of data and using the diaphragm data, but user can define other parameters)
    - `python cibrrig.postprocess.extract_coherence <session_path>`
- Compute optotagging (defaults to 473 nm wavelength and 10ms tagging window, but user can define other parameters). Uses SALT to or KS tests
    - `python cibrrig.postprocess.optotag <session_path>`



--- 
## Process video data (TODO)

