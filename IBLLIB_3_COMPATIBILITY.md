# ibllib 3.0 Compatibility Update

This update makes cibrrig compatible with ibllib 3.0 while maintaining backward compatibility with ibllib 2.x.

## Changes Made

### Dependency Updates
- Updated `ibllib` from `>=2.30, < 3.0` to `==3.13.0` (pinned version)
- Updated `spikeinterface` to `==0.103.0` (pinned version)
- Updated `kilosort[gui]` to `==4.1.0` (pinned version)
- Added `one-api==2.9.0` for proper ONE framework compatibility

### Import Updates with Fallbacks
All ibllib imports now use try/except fallback patterns to handle API changes gracefully:

1. **misc module**: `ibllib.io.misc` with fallback to `ibllib.pipes.misc`
2. **sync functions**: Updated private function names and locations
3. **plot functions**: `ibllib.plots` with fallback to `ibllib.plots.figures`
4. **QC functions**: `ibllib.ephys.qc` with fallback to `ibllib.ephys.ephysqc`
5. **apply_sync**: Multiple fallback locations for maximum compatibility

### Files Modified
- `cibrrig/archiving/ephys_data_to_alf.py`
- `cibrrig/preprocess/extract_sync_times.py`
- `cibrrig/preprocess/preproc_pipeline.py`
- `cibrrig/postprocess/convert_ks_to_alf.py`
- `cibrrig/sorting/spikeinterface_ks4.py`
- `pyproject.toml`
- `requirements.txt`

## Testing

Run the following to test the installation:
```bash
mamba activate cibrrig
pip install -e .
npx_run_all --help  # Should work without errors
```

## Backward Compatibility

The changes maintain backward compatibility by using fallback imports that will work with both ibllib 2.x and 3.x versions.