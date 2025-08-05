"""
Utility functions for working with SpikeGLX files.
Supports both compressed (.cbin) and uncompressed (.bin) formats.
"""

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


def detect_spikeglx_format(probe_dir):
    """
    Detect the format of SpikeGLX files in the probe directory.
    
    Args:
        probe_dir (Path): Path to the probe directory.
        
    Returns:
        str: File format ('.bin' for uncompressed, '.cbin' for compressed)
        
    Raises:
        FileNotFoundError: If no SpikeGLX files are found.
    """
    probe_dir = Path(probe_dir)
    
    # Check for compressed files first (cbin) - prioritize these as they're more efficient
    cbin_files = list(probe_dir.glob('*.ap.cbin')) + list(probe_dir.glob('*.lf.cbin'))
    if cbin_files:
        return '.cbin'
    
    # Check for uncompressed files (bin)
    bin_files = list(probe_dir.glob('*.ap.bin')) + list(probe_dir.glob('*.lf.bin'))
    if bin_files:
        return '.bin'
    
    # If neither found, list available files for debugging
    available_files = [f.name for f in probe_dir.glob('*')]
    raise FileNotFoundError(
        f"No SpikeGLX files (.bin or .cbin) found in {probe_dir}. "
        f"Available files: {available_files}"
    )


def find_spikeglx_files(directory, stream_type='ap', file_format=None):
    """
    Find SpikeGLX files in a directory, supporting both .bin and .cbin formats.
    
    Args:
        directory (Path): Directory to search in.
        stream_type (str): Type of stream to look for ('ap', 'lf', or 'nidq').
        file_format (str, optional): Specific format to look for ('.bin' or '.cbin').
                                   If None, searches for both and returns all found.
    
    Returns:
        list: List of found SpikeGLX files.
    """
    directory = Path(directory)
    found_files = []
    
    if file_format is None:
        # Search for both formats
        formats = ['.bin', '.cbin']
    else:
        formats = [file_format]
    
    for fmt in formats:
        if stream_type == 'nidq':
            pattern = f"*nidq{fmt}"
        else:
            pattern = f"*.{stream_type}{fmt}"
        
        found_files.extend(directory.glob(pattern))
    
    return sorted(found_files)


def detect_nidq_format(ephys_dir):
    """
    Detect the format of NIDQ files in the ephys directory.
    
    Args:
        ephys_dir (Path): Path to the ephys directory.
        
    Returns:
        str: File format ('.bin' for uncompressed, '.cbin' for compressed)
        
    Raises:
        FileNotFoundError: If no NIDQ files are found.
    """
    ephys_dir = Path(ephys_dir)
    
    # Check for compressed files first (cbin)
    cbin_files = list(ephys_dir.glob('*.nidq.cbin'))
    if cbin_files:
        return '.cbin'
    
    # Check for uncompressed files (bin)
    bin_files = list(ephys_dir.glob('*.nidq.bin'))
    if bin_files:
        return '.bin'
    
    # If neither found, list available files for debugging
    available_files = [f.name for f in ephys_dir.glob('*')]
    raise FileNotFoundError(
        f"No NIDQ files (.bin or .cbin) found in {ephys_dir}. "
        f"Available files: {available_files}"
    )