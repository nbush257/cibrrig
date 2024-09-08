"""
Code to process physiological signals.
This module is general in that it does no I/O, only processing.
Arbitrarily sourced data that has been shaped into numpy arrays can be processed
"""

import sklearn
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import scipy.signal
import scipy.interpolate


def burst_stats_dia(
    integrated, sr, dia_thresh=1, rel_height=0.9, min_distance=0.1, min_width=0.025
):
    """
    Calculate diaphragm burst features.

    Args:
        integrated (np.ndarray): Integrated diaphragm trace.
        sr (float): Sample rate of the integrated diaphragm trace.
        dia_thresh (int, optional): Prominence threshold (z-score). Defaults to 1.
        rel_height (float, optional): Relative height to use when finding onset times. Defaults to 0.9.
        min_distance (float, optional): Minimum time between breaths in seconds. Effectively sets the maximum breathing rate. Defaults to 0.1.
        min_width (float, optional): Minimum time duration to consider a breath in seconds. Defaults to 0.025.

    Returns:
        pd.DataFrame: DataFrame containing burst statistics.
    """

    # Standardize the integrated diaphragm trace
    scl = sklearn.preprocessing.StandardScaler(with_mean=0)
    integrated_scl = scl.fit_transform(integrated[:, np.newaxis]).ravel()

    # Find peaks in the standardized diaphragm trace
    pks = scipy.signal.find_peaks(
        integrated_scl,
        prominence=dia_thresh,
        distance=int(min_distance * sr),
        width=int(min_width * sr),
    )[0]

    # Calculate diaphragm onsets (lips) and offsets(rips) in samples
    lips = scipy.signal.peak_widths(integrated, pks, rel_height=rel_height)[2]
    rips = scipy.signal.peak_widths(integrated, pks, rel_height=rel_height)[3]
    lips = lips.astype("int")
    rips = rips.astype("int")

    # Remove onsets that are equal to offsets
    to_keep = rips != lips
    pks = pks[to_keep]
    lips = lips[to_keep]
    rips = rips[to_keep]

    # Find breaths fully contained in other breaths (likely sighs)
    idx = np.where(np.diff(lips) < 0)[0]
    to_rm = idx[rips[idx + 1] > rips[idx]]
    pks = np.delete(pks, to_rm)
    lips = np.delete(lips, to_rm)
    rips = np.delete(rips, to_rm)

    # Find overlapping breaths # Keep longer
    postBI = lips[1:] - rips[:-1]
    idx = np.where(postBI < 0)[0]
    dur_first = rips[idx] - lips[idx]
    dur_second = rips[idx + 1] - lips[idx + 1]
    temp = np.vstack([dur_first, dur_second]).T
    to_rm = idx + np.argmin(temp, 1)
    pks = np.delete(pks, to_rm)
    lips = np.delete(lips, to_rm)
    rips = np.delete(rips, to_rm)

    # Calculate amplitude and area under the curve for each breath
    amp = np.zeros(len(lips))
    auc = np.zeros(len(lips))
    for ii, (lip, rip) in enumerate(zip(lips, rips)):
        temp = integrated[lip:rip]
        amp[ii] = np.percentile(temp, 95)
        auc[ii] = np.trapz(temp)
    dur = rips - lips

    # Convert sample indices to seconds
    lips_t = lips / sr
    rips_t = rips / sr

    # Create a dictionary to store the burst statistics
    dia_data = {}
    dia_data["on_samp"] = lips
    dia_data["off_samp"] = rips
    dia_data["on_sec"] = lips_t
    dia_data["off_sec"] = rips_t
    dia_data["amp"] = amp
    dia_data["auc"] = auc
    dia_data["duration_sec"] = dur / sr
    dia_data["duration_samp"] = dur
    dia_data["pk_samp"] = pks
    dia_data["pk_time"] = pks / sr
    dia_data["postBI"] = np.hstack([lips_t[1:] - rips_t[:-1], [np.nan]])

    # Convert the dictionary to a DataFrame and calculate secondary burst features
    dia_df = pd.DataFrame(dia_data)
    dia_df = dia_df.eval("inst_freq=1/(duration_sec+postBI)")
    dia_df = dia_df.eval("IBI=duration_sec+postBI")
    dia_df = dia_df.query("inst_freq>0")
    dia_df = dia_df.sort_values("on_samp")

    return dia_df


def remove_EKG(x, sr, thresh=2, heartbeats=None):
    """
    Remove EKG artifacts from the diaphragm signal.
    Can either infer the heartbeat times from the signal to clean (e.g., diaphragm),
    or take an explicit set of heartbeat times
    Remove the EKG from an ephys trace using a BGM classifier

    Args:
        signal (np.ndarray): Input diaphragm signal.
        sr (float): Sampling rate of the input signal.
        thresh (float, optional): Threshold for EKG detection. Defaults to 2.
        heartbeats (np.ndarray, optional): Precomputed heartbeats. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Diaphragm signal with EKG artifacts removed.
            - np.ndarray: Detected heartbeats.
    """
    if heartbeats is not None:
        print("Removing pre-computed heartbeats")
        x_filt = _remove_EKG_explicit(x, sr, heartbeats)
    else:
        x_filt, heartbeat_samps = _remove_EKG_inferred(x, sr, thresh)
        heartbeats = heartbeat_samps / sr

    return (x_filt, heartbeats)


def _remove_EKG_explicit(x, sr, heartbeats):
    """
    Remove the EKG signal from an ephys trace using a Bayesian Gaussian Mixture (BGM) classifier.

    This function removes EKG artifacts from an electrophysiological signal by using pre-detected
    heartbeat times. It extracts segments around each heartbeat.
    It then classifies them as during a breath or not using a BGM classifier,
    It then subtracts the EKG artifact from the original signal.

    Args:
        x (np.ndarray): Signal to remove EKG from.
        sr (float): Sampling rate of the signal in samples per second.
        heartbeats (np.ndarray): Times in seconds of the detected heartbeats.
    Returns:
        np.ndarray: Signal with EKG artifacts removed.
    """

    # Convert heartbeat times to sample indices
    pks = heartbeats * sr
    pks = pks.astype("int")
    win = int(0.010 * sr)
    y = x.copy()

    # Initialize array to hold EKG segments
    ekg = np.zeros([2 * win, len(pks)])

    # Extract segments around each heartbeat
    for ii, pk in enumerate(pks):
        try:
            ekg[:, ii] = x[pk - win : pk + win]
        except Exception:
            pass

    # Calculate the standard deviation of the EKG segments
    ekg_std = np.std(ekg[:30], 0) + np.std(ekg[-30:], 0)
    ekg_std = np.log(ekg_std)
    mask = np.logical_not(np.isfinite(ekg_std))
    ekg_std[mask] = np.nanmedian(ekg_std)

    # Classify the segments using a Bayesian Gaussian Mixture (BGM) classifier
    bgm = BayesianGaussianMixture(n_components=2)
    cls = bgm.fit_predict(ekg_std[:, np.newaxis])
    cls[cls == 0] = -1
    m0 = np.nanmean(ekg_std[cls == -1])
    m1 = np.nanmean(ekg_std[cls == 1])
    # Make it so the CLS=1 is the larger standard deviation class (i.e. during a breath)
    if m0 > m1:
        cls = -cls

    # Define Savitzky-Golay filter window
    ww = int(0.0005 * sr)
    ww += ww % 2 - 1

    # Subtract the EKG artifact from the original signal
    for ii, pk in enumerate(pks):
        if pk - win < 0:
            continue
        if (pk + win) > len(y):
            continue
        if cls[ii] == -1:
            sm_ekg = scipy.signal.savgol_filter(ekg[:, ii], ww, 1)
            y[pk - win : pk + win] -= sm_ekg
        else:
            first_examp = max(ii - 5, 0)
            second_examp = min(ii + 5, y.shape[0])
            med_ekg = np.nanmedian(ekg[:, first_examp:second_examp], 1)
            y[pk - win : pk + win] -= med_ekg

    # Replace NaN values with the median of the signal
    y[np.isnan(y)] = np.nanmedian(y)
    return y


def _remove_EKG_inferred(x, sr, thresh):
    """
    Remove EKG artifacts from an ephys trace by inferring heartbeat times.

    This function applies a bandpass filter [5,500] to isolate the EKG signal, detects heartbeats,
    and then removes the EKG artifacts from the original signal.

    Args:
        x (np.ndarray): Ephys trace to clean (e.g., diaphragm or other EMG).
        sr (float): Sampling rate in samples per second.
        thresh (int, optional): Threshold in standard deviations to detect a heartbeat. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Ephys trace with the EKG filtered out. Has the same timestamps as the input trace.
            - np.ndarray: Indices of the detected heartbeats.
    """
    # Apply a bandpass filter to isolate the EKG signal
    sos = scipy.signal.butter(
        2, [5 / sr / 2, 500 / sr / 2], btype="bandpass", output="sos"
    )
    xs = scipy.signal.sosfiltfilt(sos, x)

    # Detect heartbeats based on the prominence of the filtered signal
    pks = scipy.signal.find_peaks(
        xs, prominence=thresh * np.std(xs), distance=0.05 * sr
    )[0]

    # Remove the EKG artifacts from the original signal using the detected heartbeats
    x_filt = _remove_EKG_explicit(x, sr, pks / sr)
    return (x_filt, pks)


def get_hr_from_dia(pks, dia_df, sr):
    """
    Compute heart rate from diaphragm signal.
    Not as good as getting it from a dedicated channel

    Args:
        pulse (np.ndarray): Pulse signal.
        dia_df (pd.DataFrame): DataFrame containing diaphragm burst statistics.
        sr (float): Sampling rate of the input signal.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Heart rate values.
            - np.ndarray: Detected heartbeats.
    """
    # TODO: refactor to take diaphramg onsets and offsets as input explicitly
    ons = dia_df["on_samp"]
    offs = dia_df["off_samp"]

    # Find heartbeats in between diaphragm contractions
    for on, off in zip(ons, offs):
        mask = np.logical_not(np.logical_and(pks > on, pks < off))
        pks = pks[mask]

    pulse = pd.DataFrame()
    pulse["hr (bpm)"] = 60 * sr / np.diff(pks)
    hr_smooth = pulse.rolling(50, center=True).median()
    hr_smooth.interpolate(limit_direction="both", inplace=True)
    hr_smooth["t"] = pks[:-1] / sr
    return (hr_smooth, pks / sr)


def extract_hr_from_ekg(
    x,
    sr,
    thresh=5,
    min_distance=0.05,
    low=100,
    high=1000,
    in_samples=False,
    filter=False,
):
    """
    Finds heartbeats in an EKG trace using a bandpass filter and peak detection.

    Args:
        x (np.ndarray): EKG trace.
        sr (float): Sampling rate in samples per second.
        thresh (int, optional): Prominence threshold in standard deviations to identify peaks. Defaults to 5.
        min_distance (float, optional): Minimum time between detected heartbeats in seconds. Effectively sets a maximum heart rate. Defaults to 0.05.
        low (int, optional): Low cut frequency in the bandpass filter (Hz). Defaults to 100.
        high (int, optional): High cut frequency in the bandpass filter (Hz). Defaults to 1000.
        in_samples (bool, optional): If True, return the peaks as sample indices. If False, return the peaks as time in seconds. Defaults to False.
        filter (bool, optional): If True, apply a bandpass filter to the EKG signal. Defaults to False.

    Returns:
        np.ndarray: Detected heartbeats. If `in_samples` is True, returns the sample indices of the heartbeats. Otherwise, returns the time in seconds of the heartbeats.
    """
    if filter:
        order = 8
        sos = scipy.signal.butter(
            order, [low / sr / 2, high / sr / 2], btype="bandpass", output="sos"
        )
        xs = scipy.signal.sosfiltfilt(sos, x)
    else:
        xs = x
    pks = scipy.signal.find_peaks(
        np.abs(xs), prominence=thresh * np.std(xs), distance=int(min_distance * sr)
    )[0]
    if in_samples:
        return pks
    return pks / sr


def compute_avg_hr(heartbeats, smoothing_window="10s", dt=0.1, t_target=None):
    """
    Compute the mean heart rate from detected hearteats using a sliding average window.
    First performs a median filter to remove large instantaneous outliers
    Then performs a mean filter to smooth out the average.

    Can pass t_target to interpolate to a new time basis. Otherwise defualts to a 100ms window
    If both t_target and dt are None - outputs the smoothed heart rate only with length (# heartbeats)
    Created with the help of ChatGPT3.5

    Args:
        heartbeats (np.ndarray): Timestamps of detected heartbeats.
        smoothing_window (str, optional): Smoothing window for the heart rate computation. Defaults to "10s".
        dt (float, optional): Time step for the heart rate computation. Defaults to 0.1.
        t_target (np.ndarray, optional): Target time vector for the heart rate computation. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Time vector for the average heart rate.
            - np.ndarray: Smoothed heart rate values.
    """
    if t_target is None:
        if dt is None:
            pass
        else:
            t_target = np.arange(0, heartbeats.max(), dt)

    # Convert heartbeat times to DatetimeIndex
    heartbeat_df = pd.DataFrame({"heartbeat_times": heartbeats})
    heartbeat_df["heartbeat_times"] = pd.to_datetime(
        heartbeat_df["heartbeat_times"], unit="s"
    )

    # Calculate inter-beat intervals
    heartbeat_df["inter_beat_interval"] = heartbeat_df["heartbeat_times"].diff()

    # Convert inter-beat intervals to heart rate (beats per minute)
    heartbeat_df["heart_rate"] = (
        60 / heartbeat_df["inter_beat_interval"].dt.total_seconds()
    )
    heartbeat_df.set_index("heartbeat_times", inplace=True)
    heartbeat_df["smoothed_heart_rate"] = (
        heartbeat_df["heart_rate"]
        .rolling(window="1s", center=True)
        .median()
        .rolling(smoothing_window, center=True)
        .mean()
    )

    # Interpolate the smoothed heart rate to the target time vector
    if t_target is None:
        return heartbeat_df["smoothed_heart_rate"].values
    f = scipy.interpolate.interp1d(
        heartbeats, heartbeat_df["smoothed_heart_rate"].values, fill_value="extrapolate"
    )
    smoothed_hr_out = f(t_target)

    return (t_target, smoothed_hr_out)


def compute_sighs(breath_times, auc, thresh=7, win="20s"):
    """
    Use a rolling Median Absolute Deviation (MAD) to identify sighs based on the AUC of the diaphragm.

    This function calculates the rolling MAD of the diaphragm's area under the curve (AUC) and identifies
    breaths that exceed a specified threshold as sighs.

    Args:
        breath_times (np.ndarray): Times of each breath in seconds.
        auc (np.ndarray): Area under the curve of the diaphragm for each breath.
        thresh (int, optional): Multiplier for the MAD to set the threshold for sigh detection. Defaults to 7.
        win (str, optional): Window size for the rolling MAD calculation. Defaults to '20s'.

    Returns:
        np.ndarray: Boolean array indicating which breaths are identified as sighs.
    """

    def MAD(x):
        """
        Calculate the Median Absolute Deviation (MAD) of an array.
        """
        return np.nanmedian(np.abs(x - np.nanmedian(x)))

    # Create a DataFrame to hold the AUC values and set the index to breath times
    df = pd.DataFrame()
    df["x"] = auc
    breath_times = pd.to_datetime(breath_times, unit="s")
    df.index = breath_times

    # Calculate the rolling median and MAD
    filt_breaths = df - df.rolling(win).median()
    rolling_MAD = df.rolling(window=win, center=True).apply(MAD) * thresh

    # Identify breaths that exceed the rolling MAD threshold
    idx = filt_breaths["x"] > rolling_MAD["x"]

    # Convert the index to a boolean array indicating sighs
    is_sigh = idx.values
    return is_sigh


def compute_obstructive_apneas(breath_times, inhale_onsets):
    """
    Identify obstructive apneas based on the absence of detected inhale onsets.

    This function determines periods of obstructive apnea by checking for missing inhale onsets
    while breath onsets are present.

    Args:
        breath_times (np.ndarray): Times of breath onsets (not used, but required to ensure diaphragm data is provided).
        inhale_onsets (np.ndarray): Times of inhale onsets.

    Returns:
        np.ndarray: Boolean array indicating where obstructive apneas are detected.
    """
    is_obstructive_apnea = np.isnan(inhale_onsets)
    return is_obstructive_apnea


def compute_dia_phase(
    ons, offs=None, t_start=0, t_stop=None, dt=1 / 1000, transform=True
):
    """
    Computes breathing phase based on the diaphragm signal
    Phase is [0,1] where 0 is diaphragm onset, 0.5 is diaphragm offset, and 1 is diaphragm onset again,
        - NB: Technically can generalize to any on/off signal, but standard usage should be diaphragm
        - By default, the phase is transformed to the range [-pi, pi]
    If no offset is given, the phase is linearly spaced between onsets
    If no stop time is given, the phase is computed until the last offset
    Defaults to a signal sampled at 1kHz

    Args:
        ons (np.ndarray): Onset times of the diaphragm bursts.
        offs (np.ndarray, optional): Offset times of the diaphragm bursts. Defaults to None.
        t_start (float, optional): Start time for the phase computation. Defaults to 0.
        t_stop (float, optional): Stop time for the phase computation. Defaults to None.
        dt (float, optional): Time step for the output phase signal. Defaults to 1/1000.
        transform (bool, optional): If True, transform the phase to the range [-pi, pi]. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Time vector for the phase.
            - np.ndarray: Phase values.
    """
    # Ensure that the stop time is not before the start time
    if t_stop is None:
        t_stop = offs[-1]
    if t_stop < t_start:
        raise ValueError(
            f"Stop time: {t_stop}s cannot be less than start time: {t_start}s"
        )

    # Ensure that the onsets and offsets are the same length and that the offsets are greater than the onsets
    assert len(ons) == len(offs)
    assert np.all(np.greater(offs, ons))

    # Ensure that the onset and offset times are within the start and stop times
    idx = np.logical_and(ons > t_start, offs < t_stop)
    ons = ons[idx]
    offs = offs[idx]

    # Initialize the time vector and phase signal
    t_phi = np.arange(t_start, t_stop, dt)
    phi = np.zeros_like(t_phi)

    # Compute the phase signal
    n_breaths = len(ons)
    if offs is not None:
        for ii in range(n_breaths - 1):
            on = ons[ii]
            off = offs[ii]
            next_on = ons[ii + 1]
            idx = np.searchsorted(t_phi, [on, off, next_on])
            phi[idx[0] : idx[1]] = np.linspace(0, 0.5, idx[1] - idx[0])
            try:
                phi[idx[1] : idx[2]] = np.linspace(0.5, 1, idx[2] - idx[1])
            except Exception:
                pass

    else:
        for ii in range(n_breaths - 1):
            on = ons[ii]
            next_on = ons[ii + 1]
            idx = np.searchsorted(t_phi, [on, next_on])
            phi[idx[0] : idx[1]] = np.linspace(0, 1, idx[1] - idx[0])

    # Transform the phase signal to the range [-pi, pi]
    if transform:
        phi = phi + 0.5
        phi[phi > 1] = phi[phi > 1] - 1
        phi -= 0.5
        phi *= np.pi * 2

    return (t_phi, phi)
