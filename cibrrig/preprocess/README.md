# Preprocess
---
Processing steps that happen on the raw data before sorting.

**Now supports both uncompressed (.bin) and compressed (.cbin) SpikeGLX files!**

These can either be run as command line scripts that point to a session path, or as imported function calls

###  Includes: 
 - [x] Rename file structure to ALF convention
 - [x] Extract sync signals (supports both .bin and .cbin)
 - [x] Compress video
 - [x] Extract optogenetic pulse times (supports both .bin and .cbin)
 - [x] Extract video frame times (supports both .bin and .cbin)
 - [x] Process physiology data (supports both .bin and .cbin)

### SpikeGLX File Format Support:
The preprocessing pipeline automatically detects and handles both file formats:
- **Uncompressed (.bin)**: Traditional SpikeGLX format
- **Compressed (.cbin)**: Newer compressed format (prioritized when both are present)

The pipeline will automatically:
- Detect the available file format(s)
- Log which format is being used
- Handle metadata files appropriately (.meta for .bin, .ch/.meta for .cbin)

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
