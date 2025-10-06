# Postprocess
---
Processing steps that are applied to processed data (e.g., spike times, extracted breaths)

These will happen after manual curation so we should expect these data to not live on the local computer.

## convert_ks_to_alf.py

Converts Kilosort output to ALF format and updates spike tables after manual curation in phy.

### Key Features:
- Initial export of spike sorting results to ALF format
- Update ALF tables after manual curation in phy
- Handle cluster merges (when clusters are combined in phy)
- Maintain cluster UUIDs for tracking changes
- Support for cluster labels and notes from phy

### Usage:

**Initial conversion (first time):**
```bash
convert_ks_to_alf /path/to/session kilosort4 --export-only
```

**Update after phy curation:**
```bash  
convert_ks_to_alf /path/to/session kilosort4 --update-only
```

**Both initial export and update (default):**
```bash
convert_ks_to_alf /path/to/session kilosort4
```

### Files Created/Updated:
- `clusters.uuids.csv` - Cluster UUID tracking
- `clusters.labels.csv` - Cluster labels and metadata
- `merge_info.csv` - Information about merged clusters
- Various ALF format spike files

This replaces the need for datajoint-based cluster tracking while maintaining compatibility with existing ALF workflows.
