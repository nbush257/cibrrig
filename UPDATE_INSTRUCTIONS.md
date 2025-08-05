# UPDATE INSTRUCTIONS: ibllib 3.0 Compatibility

## Quick Update (for existing users)

If you already have cibrrig installed:

```bash
# 1. Activate your environment
mamba activate cibrrig

# 2. Pull the latest changes
cd /path/to/cibrrig
git pull

# 3. Update the installation
pip install -e .

# 4. Test that it works
npx_run_all --help
```

## Fresh Installation

For new installations:

```bash
# 1. Create environment with Python 3.12
mamba create -n cibrrig python=3.12
mamba activate cibrrig

# 2. Clone and install
git clone https://github.com/nbush257/cibrrig
cd cibrrig
pip install -e .

# 3. Install phy
pip install git+https://github.com/cortex-lab/phy.git

# 4. Setup GPU for Kilosort (if needed)
pip uninstall torch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# 5. Test
python test_ibllib_compatibility.py
npx_run_all --help
```

## What Changed

- ✅ **ibllib** updated from 2.x to 3.13.0 (pinned)
- ✅ **spikeinterface** updated to 0.103.0 (pinned)  
- ✅ **kilosort** updated to 4.1.0 (pinned)
- ✅ **one-api** added at 2.9.0 (pinned)
- ✅ **Backward compatibility** maintained with fallback imports

## Troubleshooting

If you encounter import errors:

1. **Check your environment**: Make sure you're in the correct conda/mamba environment
2. **Reinstall**: Try `pip uninstall cibrrig && pip install -e .`
3. **Dependencies**: Run `pip list | grep -E "(ibllib|spikeinterface|kilosort|one-api)"` to check versions
4. **Test**: Run `python test_ibllib_compatibility.py` to diagnose issues

## For SCRI/NPX 971 Users

On the NPX computer:
```bash
cd C:/helpers/cibrrig
git pull
mamba activate cibrrig
pip install -e .
```

The main pipeline should now work with the latest versions:
```bash
npx_run_all
```