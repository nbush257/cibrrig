"""
utils.py

This module provides a collection of standard utility functions used across the project.
The functions are designed to be general-purpose with minimal dependencies on other modules.
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(level=logging.INFO)


def validate_intervals(starts, stops, overlap_ok=False):
    """
    Validates that two vectors are indeed intervals (monotonic, causal, and non-overlapping).

    Args:
        starts (np.ndarray): Array of start times.
        stops (np.ndarray): Array of stop times.
        overlap_ok (bool, optional): If True, allows overlapping intervals. Defaults to False.

    Raises:
        AssertionError: If the intervals are not monotonic, causal, or non-overlapping.
    """
    assert np.all(np.diff(starts) > 0), "Starts is not monotonic"
    assert len(starts) == len(
        stops
    ), f"Number of starts {len(starts)} does not match number of stops {len(stops)}"
    assert np.all(stops >= starts), "Stops are not all after starts"
    if not overlap_ok:
        assert np.all(starts[1:] >= stops[:-1]), "Intervals are overlapping"


def remap_time_basis(x, x_t, y_t):
    """
    Convenience function to map an analog signal x into the time basis for another signal y.

    This function maps the values of an analog signal x, which is sampled at times x_t, into the time basis of another signal y, which is sampled at times y_t.
    e.g. x is phase, y is the PCA decomposition. This allows you to get the phase value for each sample in the PCA time.

    Args:
        x (np.ndarray): Analog signal to change time basis (1D numpy array).
        x_t (np.ndarray): Time basis of the original analog signal (1D numpy array).
        y_t (np.ndarray): Time basis of the target signal (1D numpy array).

    Returns:
        np.ndarray: x mapped into the time basis of y (1D numpy array).

    Raises:
        AssertionError: If the lengths of x and x_t do not match, or if the length of the resulting index array does not match y_t.
    """
    assert len(x) == len(x_t)
    idx = np.searchsorted(x_t, y_t) - 1
    assert len(idx) == len(y_t)
    return x[idx]


def event_counts_per_epochs(event_times, starts, stops, rate=False):
    """
    Count the number of occurrences of an event in a set of intervals.

    This function counts how many times an event occurs within specified intervals.
    Optionally, it can also calculate the rate of events per interval.

    Args:
        event_times (np.ndarray): Array of event times.
        starts (np.ndarray): Array of interval start times.
        stops (np.ndarray): Array of interval stop times.
        rate (bool, optional): If True, calculate the rate of events per interval. Defaults to False.

    Returns:
        np.ndarray: Array of event counts or rates per interval.
    """
    validate_intervals(starts, stops)
    count = np.zeros(len(starts), dtype="int64")
    for ii, (start, stop) in enumerate(zip(starts, stops)):
        idx = np.logical_and(event_times > start, event_times < stop)
        count[ii] = np.sum(idx)
    if rate:
        durs = stops - starts
        return count / durs
    else:
        return count


def parse_opto_log(rr):
    """
    make a readable string from the opto log
    """
    label = rr.label.replace("opto_", "")
    try:
        np.isnan(rr.phase)
        phase = ""
    except Exception:
        phase = rr.phase + " "

    try:
        np.isnan(rr["mode"])
        mode = ""
    except Exception:
        mode = rr["mode"] + " "

    if "amplitude_mw" in rr.keys():
        amp = f"{rr.amplitude_mw:2.1f}mW "
    else:
        amp = f"{rr.amplitude:1.2f}V "

    if np.isnan(rr.frequency):
        freq = ""
    else:
        freq = f"{rr.frequency:2.0f}Hz "

    if np.isnan(rr.pulse_duration):
        pulse_dur = ""
    else:
        pulse_dur = f"{rr.pulse_duration*1000:.0f}ms"

    out = f"{label} {phase}{mode}{amp}{freq}{pulse_dur}"
    return out


def weighted_histogram(x, weights, bins, wrap=False):
    """
    Compute the weighted histogram of a variable x
    e.g.: compute PC1 as a function of respiratory phase (phi).:
    weighted_histogram(phi,PC1,np.linspace(-np.pi,np.pi),wrap=True)


    Args:
        x (_type_): Variable on the x-axis
        weights (_type_): variable on the y axis
        bins (int or array): either the number of bins on x, or explicit bin edges.
        wrap (bool, optional): whether to duplicate the first bin and add it to the end. Good for circular variables. Defaults to False.
    """
    assert x.size == weights.size
    prior, bins = np.histogram(x, bins)
    likli = []
    post, bins = np.histogram(x, weights=weights, bins=bins)
    likli = post / prior
    bins = bins[1:] - np.mean(np.diff(bins))
    if wrap:
        bins = np.concatenate([bins, [bins[0]]])
        likli = np.concatenate([likli, [likli[0]]])
    return (bins, likli)


def time_to_interval(ts, starts, stops=None, labels=None):
    """
    For each event in ts, assign it to an interval defined by starts
    Stops can be provided to close the interval. If stops is not provided
    then the interval is closed by the next event in starts

    Args:
        ts (1D numpy array): Array of event times to classify
        starts (1D numpy array): Start times of all the intervals to define
        stops (1D numpy array, optional): Stop times of all the intervals. Must be the same length as starts. Defaults to None.
        labels (1D numpy array or list, optional): Labels for each interval. Must be the same size as starts. Defaults to None.
    """
    # Sanitize starts and stops
    if stops is not None:
        validate_intervals(starts, stops)
    else:
        stops = np.concatenate([starts[1:], [np.inf]])

    # Preallocate group
    group = np.zeros(ts.shape[0], dtype="int") * np.nan
    for ii, (start, stop) in enumerate(zip(starts, stops)):
        mask = np.logical_and(ts > start, ts < stop)
        group[mask] = ii

    # Map integer group to label
    if labels is not None:
        assert len(labels) == len(starts)
        mapper = {k: v for k, v in enumerate(labels)}
        group = np.vectorize(mapper.get)(group)

    return group


def make_pre_post_trial(
    alf_object,
    intervals,
    conditions=None,
    window=None,
    pad=0,
    vars=None,
    wide=False,
    agg_func="mean",
):
    """
    Gets paired test/control data from a set of intervals (trials)
    Optionally and flexibly accepts combinatorial condition assignments for each interval
    (e.g. grid search on stimulus frequency and amplitude)

    "Comparison" is the primary output column, which indicates whether the data is from the test or control period

    Takes an alf object with a times attribute and finds all the
    observations of that variable in each interval. Then computes a
    "control" period that immediately precedes each test interval.

    Can output in "wide" form by creating a pandas pivot table where rows are trial number, split into columns for test and control

    Example use case: getting the average value of a variable before and during an opto stimulus train

    If conditions is a pandas dataframe with multiple columns, keeps the column name in the output

    Args:
        alf_object (alfio.AlfObject): ALF object containing the data.
        intervals (np.ndarray): Array of intervals (start and stop times).
        conditions (list, optional): List of conditions to filter the data. Defaults to None.
        window (tuple, optional): Time window for pre- and post-trial data. Defaults to None.
        pad (int, optional): Padding to add to the intervals. Defaults to 0.
        vars (list, optional): List of variables to include in the output. Defaults to None.
        wide (bool, optional): If True, return the data in wide format. Defaults to False.
        agg_func (str, optional): Aggregation function to use. Defaults to "mean". "count" may be useful

    Returns:
        pd.DataFrame: DataFrame containing the pre- and post-trial data.
    """
    starts = intervals[:, 0]
    stops = intervals[:, 1]

    # Create control intervals to preceed the test intervals
    window = window or stops - starts
    control_starts = intervals[:, 0] - window - pad
    control_stops = starts - pad

    if np.any(control_starts < 0):
        _log.warning(f"control starts has entries less than 0:\n{control_starts}")

    # Unpack
    assert "times" in alf_object.keys(), 'Input ALF object must have attribute "times"'
    ts = alf_object.times

    # Manipulate conditions to consider
    use_conditions = False
    if conditions is not None:
        assert (
            len(conditions) == intervals.shape[0]
        ), f"Length of conditions ({len(conditions)}) must match length of intervals ({intervals.shape[0]})"
        use_conditions = True
        if isinstance(conditions, (list, np.ndarray)):
            conditions = pd.DataFrame(conditions, columns=["condition"])
    else:
        conditions = pd.DataFrame(
            np.array(["test"] * len(starts)), columns=["condition"]
        )
    conditions = conditions.reset_index(drop=True)

    # Group conditions
    categories = conditions.columns.to_list()
    grouped = conditions.groupby(categories)

    # Initialize condition and trial arrays
    condition = pd.DataFrame(
        np.full([ts.shape[0], len(categories)], np.nan, dtype=object),
        columns=conditions.columns,
    )
    comparison = np.full_like(ts, np.nan, dtype=object)
    trial = np.full_like(ts, np.nan, dtype=float)

    # Loop over groups and assign observations to intervals
    for group_keys, group in grouped:
        _starts = starts[group.index]
        _stops = stops[group.index]
        _control_starts = control_starts[group.index]
        _control_stops = control_stops[group.index]

        # Assign observation indicies to intervals
        test_trial_num = time_to_interval(ts, _starts, _stops)
        control_trial_num = time_to_interval(ts, _control_starts, _control_stops)

        # Make sure observations are not assigned to both test and control
        overlap = np.logical_and(
            np.isfinite(test_trial_num), np.isfinite(control_trial_num)
        )
        if np.any(overlap):
            _log.warning(
                f"Observations{np.where(overlap)} are assigned to both test and control"
            )

        # Assign condition and trial numbers for test and control
        condition.loc[np.isfinite(test_trial_num), categories] = group_keys
        condition.loc[np.isfinite(control_trial_num), categories] = group_keys

        # Assign comparison and trial numbers
        comparison[np.isfinite(test_trial_num)] = "test"
        comparison[np.isfinite(control_trial_num)] = "control"

        trial[np.isfinite(test_trial_num)] = test_trial_num[np.isfinite(test_trial_num)]
        trial[np.isfinite(control_trial_num)] = control_trial_num[
            np.isfinite(control_trial_num)
        ]

    # Pandas manipulations to shape the output data
    out_df = alf_object.to_df()
    if vars:
        if not isinstance(vars, list):
            vars = [vars]
        out_df = out_df[vars]
    if use_conditions:
        out_df = out_df.merge(condition, left_index=True, right_index=True)
    out_df["trial"] = trial
    out_df["comparison"] = comparison
    out_df.dropna(axis=0, subset=["trial"], inplace=True)
    out_df["trial"] = out_df["trial"].astype("int")
    out_df = out_df.reset_index(drop=True)
    var_rename = {x: x + "_" + agg_func for x in vars}
    out_df = out_df.rename(var_rename, axis=1)
    if not use_conditions:
        categories.remove("condition")
    agg_data = (
        out_df.groupby(categories + ["comparison", "trial"]).agg(agg_func).reset_index()
    )

    if wide:
        if use_conditions:
            agg_data = pd.pivot_table(
                out_df, values=vars, columns=categories + ["comparison"], index="trial"
            )
        else:
            agg_data = pd.pivot_table(
                out_df, values=vars, columns=["comparison"], index="trial"
            )

    return agg_data


def get_pct_diff(df, vars, condition_names=None):
    """
    Calculate the percent difference between test and control conditions for specified variables.

    This function takes a long-form DataFrame output from "make_pre_post_trial" and calculates the percent difference
    and absolute difference between test and control conditions for specified variables. The DataFrame must have columns
    'trial' and 'comparison', where 'comparison' must be either 'test' or 'control'.

    Args:
        df (pd.DataFrame): Long-form pandas DataFrame containing the data. Must have columns ['trial', 'comparison'].
        condition_names (list, optional): List of columns to treat as conditions (categories). Defaults to None.
        vars (list): List of columns to treat as variables and compute percent difference on. Must be numeric.

    Returns:
        pd.DataFrame: DataFrame containing the percent difference and absolute difference for the specified variables.

    Raises:
        KeyError: If the required columns are not present in the DataFrame.

    """
    condition_names = condition_names or []

    # Separate the DataFrame into control and test conditions
    control_df = df[df["comparison"] == "control"]
    test_df = df[df["comparison"] == "test"]
    merged_df = pd.merge(
        control_df,
        test_df,
        on=condition_names + ["trial"],
        suffixes=("_control", "_test"),
    )

    # Calculate the percent difference and absolute difference for each variable
    for vv in vars:
        x = merged_df[f"{vv}_test"]
        y = merged_df[f"{vv}_control"]
        merged_df[f"{vv}_pct_diff"] = 100 * ((x - y) / y)
        merged_df[f"{vv}_diff"] = x - y

    # Define the output columns
    out_columns = (
        condition_names
        + ["trial"]
        + [f"{vv}_pct_diff" for vv in vars]
        + [f"{vv}_diff" for vv in vars]
    )

    # Create the result DataFrame with the specified output columns
    result_df = merged_df[out_columns]
    return result_df


def get_eta(x, xt, event, pre_win=0.5, post_win=None):
    """
    Compute the event-triggered average (ETA), standard deviation (std), and standard error of the mean (sem) of a covariate x.

    This function calculates the event-triggered average, standard deviation, and standard error of the mean for a given analog signal x,
    based on specified event times. It also computes the time window around each event and returns the results in a dictionary.

    Args:
        x (np.ndarray): The analog signal to compute eta for.
        xt (np.ndarray): The time vector for the analog signal x.
        event (np.ndarray): The timestamps (in seconds) of the events to average around.
        pre_win (float, optional): The window before each event to average. Defaults to 0.5 seconds.
        post_win (float, optional): The window after each event to average. Defaults to the value of pre_win.

    Returns:
        dict: A dictionary containing the following keys:
            - 'mean': The event-triggered average.
            - 'sem': The standard error of the mean.
            - 'std': The standard deviation.
            - 't': The time vector for the window around each event.
            - 'lb': The lower bound of the mean minus the sem.
            - 'ub': The upper bound of the mean plus the sem.

    Raises:
        AssertionError: If the lengths of xt and x do not match.
    """
    assert len(xt) == len(x)
    if post_win is None:
        post_win = pre_win

    # Infer sampling rate
    dt = xt[1] - xt[0]

    # Identify the samples corresponding to each event
    samps = np.searchsorted(xt, event)

    # Preallocate the event-triggered array where each column is an event
    win_samps_pre = int(pre_win / dt)
    win_samps_post = int(post_win / dt)
    event_triggered = np.zeros([win_samps_pre + win_samps_post, len(samps)])

    # Fill the event-triggered array
    for ii, samp in enumerate(samps):
        if (samp - win_samps_pre) < 0:
            continue
        if (samp + win_samps_post) > len(x):
            continue
        event_triggered[:, ii] = x[samp - win_samps_pre : samp + win_samps_post]

    # Compute the event-triggered average, standard deviation, and standard error of the mean
    st_average = np.nanmean(event_triggered, 1)
    st_sem = np.nanstd(event_triggered, 1) / np.sqrt(len(samps))
    st_std = np.nanstd(event_triggered, 1)
    win_t = np.linspace(-pre_win, post_win, (win_samps_pre + win_samps_post))
    lb = st_average - st_sem
    ub = st_average + st_sem

    # Return the results in a dictionary
    eta = {
        "mean": st_average,
        "sem": st_sem,
        "std": st_std,
        "t": win_t,
        "lb": lb,
        "ub": ub,
    }

    return eta


def get_good_spikes(spikes, clusters):
    """
    Convenience function to return only good spikes.

    This function filters out spikes that belong to clusters marked as "bad" based on the 'bitwise_fail' metric.
    It returns only the spikes that belong to clusters passing the quality control.

    Args:
        spikes (AlfBunch): ALF object containing spike data, with keys such as 'times' and 'clusters'.
        clusters (AlfBunch): ALF object containing cluster metrics, including a 'bitwise_fail' column.

    Returns:
        tuple: A tuple containing:
            - spikes (AlfBunch): Filtered ALF object containing only good spikes.
            - cluster_ids (np.ndarray): Array of cluster IDs that passed the quality control.
    """
    # Get the cluster IDs that pass the quality control
    cluster_ids = clusters.metrics.query("bitwise_fail==0")["cluster_id"].values

    # Create a boolean index for spikes belonging to good clusters
    idx = np.isin(spikes.clusters, cluster_ids)

    # Filter the spikes to include only those belonging to good clusters
    for k in spikes.keys():
        spikes[k] = spikes[k][idx]
    return (spikes, cluster_ids)
