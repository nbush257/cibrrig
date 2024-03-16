# CIBR RIG
---
## Usage
Code to integrate the hardware and software on the Neuropixle rig in JMB 971 at Seattle Childrens Research Institute, Center for Integrative Brain Research (SCRI-CIBR)
This code is maintained by Nick Bush in the Ramirez Lab and is subject to change.

The rig is designed to monitor breathing and behavior in a head-fixed mouse while recording from neuropixels throughout the brain. Rig is capable of hot-swapp


**Updated 2024-02-23**

---
## Installation
TODO

Helper packages (Packages from other groups (e.g., kilosort)) should live in `C:/helpers` on the NPX computer so they are available to all users

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
## Primary pipeline

 - Backup 
    ```
    cd archiving
    python backup.py archive <run_path>
    ```
    - Follow GUI steps
- Rename
    `python ephys_data_to_alf.py <run_path>`
 - Preprocess:
    ```
    cd preprocess
    python pipeline <awake/anesthetized> <session_path>
    ```
- Sort
    ```
    cd sort
    python -W ignore spikeinterface_ks4.py <session_path>
    ```
- Move to remote
    ```
    cd archiving
    python backup.py working 
    ```
    - Follow GUI steps


- Extract coherence and optotagging (TODO: Now that data structure is clean - optotagging may be better implemetned)
    ```
    cd postprocess
    python extract_coherence.py <session_path> <kwargs>
    python opto.py <phy_path> <opto_fn> <log_fn> <kwargs>
    ```
- Process video data (TODO)

