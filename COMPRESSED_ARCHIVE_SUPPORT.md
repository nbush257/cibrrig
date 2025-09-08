# Compressed Archive Pipeline Support

This document describes the new compressed archive functionality implemented in the cibrrig pipeline.

## Overview

The pipeline now supports compressing electrophysiological data locally before archiving, allowing processing directly from compressed `.cbin` files. This reduces network transfer times and storage requirements while maintaining full functionality.

## New Workflow

### Before (Legacy):
1. Copy uncompressed data to archive 
2. Compress data on archive server
3. Process locally from uncompressed files
4. Move processed data to working directory

### After (New Default):
1. **Compress data locally** (.bin → .cbin)
2. **Copy compressed archive** to backup location
3. **Process pipeline** from compressed .cbin files
4. Move processed data to working directory

## Usage

### Command Line Interface

#### Main Pipeline
```bash
# Use new compressed workflow (default)
npx_run_all_no_gui /path/to/data /path/to/working /path/to/archive

# Use legacy remote compression  
npx_run_all_no_gui /path/to/data /path/to/working /path/to/archive --no_local_compression
```

#### Backup Only
```bash
# Use new local compression (default)
backup /path/to/local/data /path/to/archive

# Use legacy remote compression
backup /path/to/local/data /path/to/archive --no_local_compression
```

### Python API

```python
from cibrrig.main_pipeline import run
from cibrrig.archiving.backup import no_gui

# New compressed workflow (default)
run(
    local_path, 
    working_path, 
    archive_path, 
    remove_opto_artifact=False, 
    run_ephysQC=True,
    compress_locally=True  # Default
)

# Legacy workflow  
run(
    local_path, 
    working_path, 
    archive_path, 
    remove_opto_artifact=False, 
    run_ephysQC=True,
    compress_locally=False
)

# Backup with local compression
no_gui(local_path, archive_path, compress_locally=True)
```

## Benefits

- **Faster backups**: Compressed files transfer faster over network
- **Reduced storage**: Archive requires less disk space  
- **Same performance**: spikeinterface processes .cbin files as efficiently as .bin files
- **Backward compatible**: Legacy workflow still available

## Compatibility

- **File formats**: Supports both `.bin` and `.cbin` files throughout pipeline
- **Mixed environments**: Can process sessions with mix of compressed/uncompressed files
- **Existing scripts**: Work unchanged (new behavior is default but transparent)
- **Tools**: Compatible with spikeinterface, ibl-neuropixels, and all preprocessing tools

## Technical Details

### Components Updated

1. **backup.py**
   - `compress_ephys_files_local()` - Compress files before archiving
   - `full_archive_with_local_compression()` - New workflow
   - `no_gui(compress_locally=True)` - API parameter

2. **main_pipeline.py**  
   - `run(compress_locally=True)` - API parameter
   - Updated CLI with `--no_local_compression` flag
   - Modified workflow order

3. **File pattern compatibility**
   - `alf_utils.py` - Searches `*nidq.bin` and `*nidq.cbin`
   - `ephys_data_to_alf.py` - Processes `*ap.bin` and `*ap.cbin`  
   - `synchronize_sorting_to_aux.py` - Handles both file types

### Default Behavior

- **New installations**: Use compressed workflow by default
- **Existing scripts**: Work unchanged, get new behavior automatically  
- **Legacy mode**: Available via `--no_local_compression` flags

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_compressed_archive.py -v
```

Tests cover:
- New method functionality
- Parameter propagation  
- Backward compatibility
- File type support

## Migration

No migration required! The new functionality is:

- **Default enabled**: New behavior is the default
- **Transparent**: Existing code works unchanged  
- **Optional**: Legacy behavior available via flags
- **Gradual**: Can migrate systems individually

## Troubleshooting

### Issue: Pipeline fails to find data files
**Solution**: Ensure spikeglx version supports .cbin files (most recent versions do)

### Issue: Need legacy compression behavior  
**Solution**: Use `--no_local_compression` flag

### Issue: Mixed .bin/.cbin files in session
**Solution**: Pipeline handles mixed environments automatically

## Support

This implementation maintains full backward compatibility while providing the requested compressed archive functionality from issue #22.