# Preprocess
---
Processing steps that happen on the raw data before sorting.

These can either be run as command line scripts that point to a session path, or as imported function calls

###  Includes: 
 - [x] Rename file structure to ALF convention
 - [x] Extract sync signals
 - [x] Compress video
 - [x] Extract optogenetic pulse times
 - [x] Extract video frame times
 - [x] Process physiology data

### Usage:
Sync extractors work on a "session", which is equivalent to a "gate". All recordings in one session should be sorted to gether.

Make sure data has first been backed up (frozen) with `archiving.backup.py`
### Pipeline (order matters):
 1. `ephys_data_to_alf.py <session_path>`
 1. `extract_sync_times.py <session_path>`
 1. `extract_frame_times.py <session_path>`
 1. `extract_opto_times.py <session_path>`
 1. `extract_physiology.py <session_path> <args>`

Then go on to sorting.
