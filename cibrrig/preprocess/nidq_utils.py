"""
I/O functions with NIDAQ data specifically for some of our physiology needs.

This module is primarily focused on loading and manipulating data from
the NIDAQ files recorded by spikeglx, and passes most computation to the physiology module,
which is more general in scope.

Supports both uncompressed (.bin) and compressed (.cbin) SpikeGLX files.
"""

import scipy.integrate
import numpy as np
import scipy.signal as sig
from scipy.ndimage import median_filter
import warnings
import spikeglx

try:
    from . import physiology
    from ..utils.spikeglx_utils import find_spikeglx_files
except ImportError:
    import sys
    sys.path.append("../")
    import physiology
    # For standalone usage, import from relative path
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.spikeglx_utils import find_spikeglx_files
import logging
import re
import matplotlib.pyplot as plt

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def get_triggers(session_path):
    """
    Looks through all the NIDQ files to extract the trigger strings.

    Args:
        session_path (Path): Path to the session directory.

    Returns:
        list: Sorted list of trigger strings found in the NIDQ files.
    """
    # Find both .bin and .cbin NIDQ files
    ni_files = find_spikeglx_files(session_path.joinpath("raw_ephys_data"), 'nidq')
    trig_strings = [get_trig_string(x.stem) for x in ni_files]
    trig_strings.sort()
    return trig_strings


def get_trig_string(in_str):
    """
    Extract the trigger string from a given input string using regex.

    Args:
        in_str (str): Input string containing the trigger information.

    Returns:
        str: Extracted trigger string.
    """
    trig_string = re.search("t\d{1,3}", in_str).group()
    return trig_string


def _extract_ds_chan(SR, chan_id, ds_factor=10):
    """
    Extract and downsample a specific analog channel from the SpikeGLX reader.

    Args:
        SR (spikeglx.Reader): SpikeGLX reader object for the recording.
        chan_id (int): Channel ID to extract.
        ds_factor (int, optional): Downsampling factor. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Downsampled data from the specified channel.
            - float: Downsampled sampling rate.
    """
    assert type(ds_factor) is int
    sr = SR.fs
    sr_sub = sr / ds_factor
    dat = SR.read(nsel=slice(None, None, ds_factor), csel=chan_id, sync=False)
    return (dat, sr_sub)


def load_mmap(fn):
    """
    Load a memory-mapped Nidaq file.

    Args:
        fn (Path): Path to the Nidaq.bin file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Memory-mapped data array.
            - dict: Metadata dictionary.
    """
    SR = spikeglx.Reader(fn)
    mmap = SR.read()[0]
    return (mmap, SR.meta)


def binary_onsets(x, thresh):
    """
    Binarize a signal at the level "thresh" and return the onset and offset indices.

    Args:
        x (np.ndarray): Input signal.
        thresh (float): Threshold value to determine binary state of HIGH (1) or LOW (0).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Indices of onset samples.
            - np.ndarray: Indices of offset samples.

    Raises:
        ValueError: If the number of onsets does not match the number of offsets.
    """

    # Convert the signal to a boolean array based on the threshold
    xbool = x > thresh
    # Find the onsets and offsets
    ons = np.where(np.diff(xbool.astype("int")) == 1)[0]
    offs = np.where(np.diff(xbool.astype("int")) == -1)[0]

    # Deal with edge cases
    if xbool[0]:
        offs = offs[1:]
    if xbool[-1]:
        ons = ons[:-1]
    if len(ons) != len(offs):
        plt.plot(x)
        plt.axhline(thresh)
        raise ValueError("Onsets does not match offsets")
    return (ons, offs)


def get_tvec(dat, sr):
    """
    Generate a time vector for a given data array and sampling rate.


    Args:
        dat (np.ndarray): Data array.
        sr (float): Sampling rate.

    Returns:
        np.ndarray: Time vector.
    """
    assert len(dat.shape) == 1, "Input data 'dat' must be one-dimensional."
    tvec = np.linspace(0, len(dat) / sr, len(dat))
    return tvec


def get_tvec_from_fn(fn):
    """
    Generate a time vector from a Nidaq file.

    Args:
        fn (Path): Path to the Nidaq file.

    Returns:
        np.ndarray: Time vector corresponding to the data in the file.
    """
    SR = spikeglx.Reader(fn)
    tvec = get_tvec_from_SR(SR)
    return tvec


def get_tvec_from_SR(SR):
    """
    Generate a time vector from a SpikeGLX reader object.

    Args:
        SR (spikeglx.Reader): SpikeGLX reader object for the recording.

    Returns:
        np.ndarray: Time vector corresponding to the data in the reader object.
    """
    sr = SR.fs
    n_samps = SR.ns
    tvec = np.linspace(0, n_samps / sr, n_samps)
    return tvec


def load_ds_pdiff(SR, chan_id, ds_factor=10, inhale_dir=-1):
    """
    Load and downsample the pdiff (differential pressure sensor) data.

    Args:
        SR (spikeglx.Reader): SpikeGLX reader object for the recording.
        chan_id (int): Channel ID for the pdiff signal.
        ds_factor (int, optional): Downsampling factor. Defaults to 10.
        inhale_dir (int, optional): Direction of inhalation. Defaults to -1.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Downsampled pdiff data.
            - float: Downsampled sampling rate.
    """
    dat, sr_sub = _extract_ds_chan(SR, chan_id, ds_factor)
    dat = dat * inhale_dir

    # Do not do any baseline correction on the PDIFF because it is AC.
    return (dat, sr_sub)


def load_dia_emg(SR, chan_id):
    """
    Read the raw diaphragm EMG data. Does not downsample the data

    Subtract the mean from the raw data.

    Args:
        SR (spikeglx.Reader): SpikeGLX reader object for the recording.
        chan_id (int): Channel ID for the diaphragm EMG signal.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Raw diaphragm EMG data.
            - float: Sampling rate of the diaphragm recording.
    """
    ds_factor = 1  # Do no downsampling here
    dat, sr = _extract_ds_chan(SR, chan_id, ds_factor)
    dat = dat - np.mean(dat)
    return (dat, sr)


def filt_int_ds_dia(x, sr, ds_factor=10, rel_height=0.95, heartbeats=None):
    """
    Filter, integrate, and downsample the diaphragm EMG signal. Detect and summarize the diaphragm bursts.
    Uses median filtering to smooth the signal, which can be slow but is effective.

    Args:
        x (np.ndarray): Raw diaphragm EMG signal.
        sr (float): Sampling rate of the input signal.
        ds_factor (int, optional): Downsampling factor. Defaults to 10.
        rel_height (float, optional): Relative height for burst detection. Defaults to 0.95.
        heartbeats (np.ndarray, optional): Precomputed heartbeats. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with burst statistics.
            - np.ndarray: Downsampled and normalized diaphragm signal.
            - float: Downsampled sampling rate.
            - np.ndarray: Heart rate data.
            - np.ndarray: Filtered diaphragm signal.
            - np.ndarray: Detected heartbeats.
    """
    assert type(ds_factor) is int

    # Remove the EKG artifact
    _log.info("Removing the EKG...")
    dia_filt, pulse = physiology.remove_EKG(x, sr, thresh=2, heartbeats=heartbeats)
    dia_filt[np.isnan(dia_filt)] = np.nanmedian(dia_filt)

    # Filter for high frequency signal
    sos = sig.butter(2, [300 / sr / 2, 5000 / sr / 2], btype="bandpass", output="sos")
    dia_filt = sig.sosfilt(sos, dia_filt)

    # Use medfilt to get the smoothed rectified EMG
    _log.info("Smoothing the rectified trace...")
    window_length = int(0.05 * np.round(sr)) + 1
    if window_length % 2 == 0:
        window_length += 1
    dd = median_filter(np.abs(dia_filt), window_length)

    # Smooth it out a little more
    window_length = int(0.01 * np.round(sr)) + 1
    if window_length % 2 == 0:
        window_length += 1
    dia_smooth = sig.savgol_filter(dd, window_length=window_length, polyorder=1)

    # Downsample because we don't need this at the original smapling rate
    dia_sub = dia_smooth[::ds_factor]
    sr_sub = sr / ds_factor

    # Get the burst statistics
    warnings.filterwarnings("ignore")
    dia_df = physiology.burst_stats_dia(dia_sub, sr_sub, rel_height=rel_height)
    warnings.filterwarnings("default")

    # Compute heart rate from diaphragm signal if heartbeats are not provided
    HR = None
    if heartbeats is None:
        HR, heartbeats = physiology.get_hr_from_dia(pulse / ds_factor, dia_df, sr_sub)

    # Normalize the integrated diaphragm to a z-score.
    dia_df["amp_z"] = dia_df["amp"] / np.std(dia_sub)
    dia_sub = dia_sub / np.std(dia_sub)
    _log.info("Done processing diaphragm")

    return (dia_df, dia_sub, sr_sub, HR, dia_filt, heartbeats)


def extract_hr_channel(SR, ekg_chan=2):
    """
    Extract heart rate from a dedicated EKG channel.
    First subtracts the mean from the EKG signal.

    Passes the mean-subtracted EKG signal to the physiology.extract_hr_from_ekg function.

    Args:
        SR (spikeglx.Reader): SpikeGLX reader object for the recording.
        ekg_chan (int, optional): Channel ID for the EKG signal. Defaults to 2.

    Returns:
        np.ndarray: Timestamps of detected heartbeats.

    Raises:
        ValueError: If the EKG signal cannot be processed.
    """
    ds_factor = 1  # No downsampling here
    dat, sr = _extract_ds_chan(SR, ekg_chan, ds_factor=ds_factor)
    dat = dat - np.mean(dat)

    heartbeats = physiology.extract_hr_from_ekg(dat, sr)

    return heartbeats


def extract_temp(SR, temp_chan=7, ds_factor=10):
    """
    Extract the temperature from the FHC DC temp controller. Assumes the manufacturer's calibration.

    Args:
        SR (spikeglx.Reader): SpikeGLX reader object for the recording.
        temp_chan (int, optional): Channel ID for the temperature signal. Defaults to 7.
        ds_factor (int, optional): Downsampling factor. Defaults to 10.

    Returns:
        np.ndarray: Downsampled temperature data.
    """
    assert type(ds_factor) is int
    dat, sr = _extract_ds_chan(SR, temp_chan, ds_factor=ds_factor)
    # 0v=25C, 2V = 45C, 100mv=1C
    vout_map = [0, 2]
    temp_map = [25, 45]
    temp_f = scipy.interpolate.interp1d(vout_map, temp_map)
    temp_out = temp_f(dat)
    temp_out = scipy.signal.savgol_filter(temp_out, 101, 1)
    return temp_out


def filt_int_ds_arbitrary(x, sr, ds_factor=10):
    """
    Filter, integrate, and downsample an arbitrary signal.

    Applies a second order Butterworth bandpass filter between 300 and 5000 Hz.

    Args:
        x (np.ndarray): Input signal.
        sr (float): Sampling rate of the input signal.
        ds_factor (int, optional): Downsampling factor. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Processed and downsampled signal.
            - float: Downsampled sampling rate.
            - np.ndarray: Filtered signal.
    """
    assert type(ds_factor) is int

    # Filter for high frequency signal
    sos = sig.butter(2, [300 / sr / 2, 5000 / sr / 2], btype="bandpass", output="sos")
    x_filt = sig.sosfilt(sos, x)

    # Use medfilt to get the smoothed rectified EMG
    _log.info("Smoothing the rectified trace...")
    window_length = int(0.05 * np.round(sr)) + 1
    if window_length % 2 == 0:
        window_length += 1
    dd = median_filter(np.abs(x_filt), window_length)

    # Smooth it out a little more
    window_length = int(0.01 * np.round(sr)) + 1
    if window_length % 2 == 0:
        window_length += 1
    dia_smooth = sig.savgol_filter(dd, window_length=window_length, polyorder=1)

    # Downsample because we don't need this at the original smapling rate
    x_sub = dia_smooth[::ds_factor]
    sr_sub = sr / ds_factor

    # Normalize the integrated diaphragm to a z-score.
    x_sub = x_sub / np.std(x_sub)
    _log.info("Done processing signal")
    return (x_sub, sr_sub, x_filt)
