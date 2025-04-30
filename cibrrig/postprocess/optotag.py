"""
Functions to work with optogenetic stimulations, and in particular, tags.
When run from command line will perform tag statistics
"""

import numpy as np
import scipy.io.matlab as sio
from pathlib import Path
import pandas as pd
from brainbox import singlecell
import subprocess
import click
import matplotlib.pyplot as plt
import seaborn as sns
import one.alf.io as alfio
from tqdm import tqdm
import sys
import scipy.stats

WAVELENGTH_COLOR = {
    635: "#ff3900",
    473: "#00b7ff",
}  # Dictionary that maps a wavelength to a hex

# Requirements for is_tagged=True. All criteria must be met
SALT_P_CUTOFF = 0.001  # Consider units p<SALT_P_CUTOFF as tagged
MIN_PCT_TAGS_WITH_SPIKES = (
    33  # Consider units with spikes on at least this percentage of stims as tagged
)
RATIO = 3  # Spike rates must increase by this factor to be considered tagged. (e.g. RATIO=3, pre_FR=2. then post_FR>= 6 to be considered tagged)


def compute_pre_post_raster(
    spike_times,
    spike_clusters,
    cluster_ids,
    stim_times,
    stim_duration=None,
    window_time=0.5,
    bin_size=0.001,
    mask_dur=0.002,
):
    """
    Creates the rasters pre and post stimulation time.
    Optionally blanks periods around onset and offset of light to zero (default behavior)
    Wraps to the IBL bin_spikes2D

    Args:
        spike_times (np.ndarray): Array of times in seconds for each spike.
        spike_clusters (np.ndarray): Array of clusters for each spike.
        cluster_ids (np.ndarray): List of clusters to include.
        stim_times (np.ndarray): Onset times of each opto stim to align to.
        stim_duration (float): Duration of stimulus in seconds.
        window_time (float, optional): Size of the PETH to compute in seconds. Defaults to 0.5.
        bin_size (float, optional): Size of the bins of the PETH in seconds. Defaults to 0.001.
        mask_dur (float, optional): Duration to mask to zero near onset and offset in seconds. Defaults to 0.002.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: pre_raster - Raster of spike counts before stimulus onset. Size: [n_stims, n_units, n_timebins].
            - np.ndarray: post_raster - Raster of spike counts after stimulus onset. Size: [n_stims, n_units, n_timebins].
    """
    pre_raster, pre_tscale = singlecell.bin_spikes2D(
        spike_times,
        spike_clusters,
        cluster_ids,
        align_times=stim_times,
        pre_time=window_time + mask_dur,
        post_time=-mask_dur,  # Calculate until 2 ms before the stimulus
        bin_size=bin_size,
    )

    post_raster, post_tscale = singlecell.bin_spikes2D(
        spike_times,
        spike_clusters,
        cluster_ids,
        align_times=stim_times,
        pre_time=-mask_dur,  # ignore 2ms after stimulus onset to avoid artifactual spikes
        post_time=window_time + mask_dur,
        bin_size=bin_size,
    )

    # if stim_duration exists, remove any spikes within - 1 ms and + mask_duration of offset time
    if stim_duration is not None:  
        stim_offsets_samp = np.searchsorted(post_tscale, stim_duration)
        post_raster[
            :, :, stim_offsets_samp - 1 : stim_offsets_samp + int(mask_dur / bin_size)
        ] = 0

    return (pre_raster, post_raster)


def run_salt(
    spike_times,
    spike_clusters,
    cluster_ids,
    stim_times,
    window_time=0.5,
    stim_duration=None,
    consideration_window=0.01,
):
    """
    Runs the Stimulus Associated Latency Test (SALT - See Kvitsiani 2013)
    on given units.

    Must pass data to matlab, and does so via saving to a temporary mat file.
    Automatically deletes the temporary mat file.

    Args:
        spike_times (np.ndarray): Array of times in seconds for each spike.
        spike_clusters (np.ndarray): Array of clusters for each spike.
        cluster_ids (np.ndarray): List of clusters to include.
        stim_times (np.ndarray): Onset times of each opto stim to align to.
        window_time (float, optional): Size of the PETH to compute in seconds. Defaults to 0.5.
        stim_duration (float, optional): Duration of stimulus in seconds. Defaults to None.
        consideration_window (float, optional): Time window to consider for analysis in seconds. Defaults to 0.01.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: p_stat - SALT p-values for each cluster.
            - np.ndarray: I_stat - SALT I-values for each cluster.
    """

    pre_raster, post_raster = compute_pre_post_raster(
        spike_times,
        spike_clusters,
        cluster_ids,
        stim_times,
        window_time=window_time,
        stim_duration=stim_duration,
        bin_size=0.001,
        mask_dur=0.002,
    )

    # Sanitize raster
    if np.any(pre_raster < 0) | np.any(post_raster < 0):
        print("Warning: spike counts less than zero found. Setting to 0")
        pre_raster[pre_raster < 0] = 0
        post_raster[post_raster < 0] = 0

    if np.any(pre_raster > 1) | np.any(post_raster > 1):
        print(
            f"Warning: Multiple spikes in a single ms bin found (max is {max(np.max(pre_raster),np.max(post_raster))}). Truncating to 1"
        )
        pre_raster[pre_raster > 1] = 1
        post_raster[post_raster > 1] = 1

    dat_mat = {}
    dat_mat["pre_raster"] = pre_raster
    dat_mat["post_raster"] = post_raster
    dat_mat["cluster_ids"] = cluster_ids
    dat_mat["consideration_window"] = consideration_window
    sio.savemat(".temp_salt.mat", dat_mat)

    command = ["matlab", "-batch", "python_SALT('.temp_salt.mat');"]
    subprocess.run(command, check=True)

    salt_rez = sio.loadmat(".temp_salt.mat")
    Path(".temp_salt.mat").unlink()
    p_stat = salt_rez["p_stat"]
    I_stat = salt_rez["I_stat"]

    return p_stat, I_stat


def compute_tagging_summary(
    spike_times, spike_clusters, cluster_ids, stim_times, stim_duration=None,window_time=0.01
):
    """
    Computes the number of stims that a spike was observed and the pre and post stim spike rate

    Args:
        spike_times (1D numpy array): times in  seconds of every spike
        spike_clusters (1D numpy array): cluster assignments of every spike
        cluster_ids (1D numpy array): cluster ids to use.
        stim_times (1D numpy array): onset times of the optogenetic stimulus
        window_time (float, optional): window to compute spike rate in seconds. Defaults to 0.01.
    Returns:
        n_responsive_stims (1D numpy array): Number of stimulations that evoked at least one spike
        pre_spikerate (1D numpy array): Spike rate in the window before stimulus onset
        post_spikerate (1D numpy array): Spike rate in the windw after stimulus onset
    """
    bin_size = 0.001
    n_stims = stim_times.shape[0]
    pre_spikecounts, post_spikecounts = compute_pre_post_raster(
        spike_times,
        spike_clusters,
        cluster_ids,
        stim_times,
        stim_duration=stim_duration,
        window_time=window_time,
        bin_size=bin_size,
    )
    pre_spikecounts = pre_spikecounts.sum(2)
    post_spikecounts = post_spikecounts.sum(2)

    n_responsive_stims = np.sum(post_spikecounts.astype("bool"), 0)
    pre_spikerate = np.sum(pre_spikecounts, 0) / n_stims / window_time
    post_spikerate = np.sum(post_spikecounts, 0) / n_stims / window_time

    return (n_responsive_stims, pre_spikerate, post_spikerate)


def extract_tagging_from_logs(log_df, laser, verbose=True):
    """Finds the optogenetic stimulations that are associated with a logged tagging episode.

    Args:
        log_df (pandas.DataFrame): The log from the experiment, should be an autogenerated TSV.
        opto_df (pandas.DataFrame): The optogenetic stimulation dataframe extracted from the analog trace. Note: it must be synchronized.
        verbose (bool, optional): If True, prints verbose output. Defaults to True.

    Raises:
        NotImplementedError: If more than one tagging episode is found.
        ValueError: If no tagging episodes are found.

    Returns:
        pandas.DataFrame: A subdataframe from opto_df that only contains the tagging data.
    """

    # Extract the tagging start and end
    tag_epoch = log_df.query('label == "opto_tagging"')
    tag_starts = tag_epoch["start_time"].values
    tag_ends = tag_epoch["end_time"].values

    # Handle if more than one or less than 1 tagging episodes were found
    if len(tag_starts) > 1:
        raise NotImplementedError(
            "More than one tagging episode has not been implemetned yet"
        )
    elif len(tag_starts) == 0:
        raise ValueError("No Tagging episodes found.")
    else:
        pass

    # Subset (add a little buffer time so as not to miss first or last stim)
    tag_start = tag_starts[0] - 0.5
    tag_end = tag_ends[0] + 0.5

    # slice
    idx = np.logical_and(
        laser.intervals[:, 0] > tag_start, laser.intervals[:, 1] < tag_end
    )
    tags = alfio.AlfBunch()
    for k in laser.keys():
        tags[k] = laser[k][idx]
    dur_mean = np.diff(tags.intervals, 1).mean()

    # Verbose and return
    print(
        f"Found {tags.intervals.shape[0]} tag stimulations with average duration {dur_mean:0.02}s"
    ) if verbose else None
    return tags


def make_plots(
    spike_times,
    spike_clusters,
    cluster_ids,
    tags,
    save_folder,
    optotag_rez=None,
    pre_time=None,
    post_time=None,
    wavelength=473,
    consideration_window=0.01,
    cmap="magma",
    plot_desc=True,
    method="salt",
):
    """
    Plots rasters and PETHs for each cell aligned to stimulus.
    Optionally marks each plot with some statistics.
    Saves to both png and svg

    Args:
        spike_times (np.ndarray): Array of times in seconds for each spike.
        spike_clusters (np.ndarray): Array of clusters for each spike.
        cluster_ids (np.ndarray): List of clusters to include.
        tags (np.ndarray): Array of tags for each stimulus.
        save_folder (Path): Folder to save the plots.
        tag_df (pd.DataFrame, optional): DataFrame containing tagging results. Defaults to None.
        pre_time (float, optional): Time before stimulus onset to include in plots. Defaults to None.
        post_time (float, optional): Time after stimulus onset to include in plots. Defaults to None.
        wavelength (int, optional): Wavelength of the stimulus in nm. Defaults to 473.
        consideration_window (float, optional): Time window to consider for analysis in seconds. Defaults to 0.01.
        cmap (str, optional): Colormap to use for plots. Defaults to "magma".
        plot_desc (bool, optional): Whether to include descriptive text in plots. Defaults to False.
    """
    pre_time = pre_time or consideration_window * 2
    post_time = post_time or consideration_window * 2
    if not save_folder.exists():
        save_folder.mkdir()
    else:
        print("Removing old figures.")
        for fn in save_folder.glob("*.png"):
            fn.unlink()
        for fn in save_folder.glob("*.svg"):
            fn.unlink()
    stim_duration = np.diff(tags.intervals, 1).mean()
    stim_times = tags.intervals[:, 0]
    n_stims = stim_times.shape[0]
    if wavelength == 635:
        bin_size = 0.005
    else:
        bin_size = 0.0025
    peths_fine, raster_fine = singlecell.calculate_peths(
        spike_times,
        spike_clusters,
        cluster_ids,
        stim_times,
        pre_time=pre_time,
        post_time=stim_duration + post_time,
        bin_size=0.001,
        smoothing=0,
    )

    peths, _ = singlecell.calculate_peths(
        spike_times,
        spike_clusters,
        cluster_ids,
        stim_times,
        pre_time=pre_time,
        post_time=stim_duration + post_time,
        bin_size=bin_size,
        smoothing=0,
    )

    stim_no, clu_id, sps = np.where(raster_fine)
    spt = peths_fine["tscale"][sps]

    for ii, clu in enumerate(tqdm(cluster_ids, desc="Making plots")):
        plt.close("all")

        # Set up plot
        f, ax = plt.subplots(nrows=2, figsize=(4, 4), sharex=True)

        # Plot data
        ax[0].vlines(
            spt[clu_id == ii],
            stim_no[clu_id == ii] - 0.25,
            stim_no[clu_id == ii] + 0.25,
            color="k",
            lw=1,
        )
        ax[1].plot(peths["tscale"], peths["means"][ii], color="k")
        lb = peths["means"][ii] - peths["stds"][ii] / np.sqrt(n_stims)
        ub = peths["means"][ii] + peths["stds"][ii] / np.sqrt(n_stims)
        ax[1].fill_between(peths["tscale"], lb, ub, alpha=0.3, color="k")

        # Plot stim limits
        for aa in ax:
            aa.axvspan(0, stim_duration, color=WAVELENGTH_COLOR[wavelength], alpha=0.25)
            aa.axvline(0, color="c", ls=":", lw=1)
            aa.axvline(stim_duration, color="c", ls=":", lw=1)
            aa.axvline(consideration_window, color="k", ls=":", lw=1)

        # Formatting
        ax[0].set_ylim([0, n_stims])
        ax[0].set_yticks([0, n_stims])
        ax[0].set_ylabel("Stim #")
        ax[1].set_ylabel("F.R. (sp/s)")
        ax[1].set_xlabel("Time (s)")
        ax[0].set_title(f"Cluster {clu}")

        # Additional info if available
        if optotag_rez is not None:
            this_cell = optotag_rez.query("cluster_id==@clu")

        if optotag_rez is not None and plot_desc:
            p_salt = this_cell["salt_p_stat"].values[0]
            p_ks = this_cell["p_ks"].values[0]
            base_rate = this_cell["base_rate"].values[0]
            stim_rate = this_cell["stim_rate"].values[0]
            ax[0].text(
                0.8,
                0.8,
                f"{p_ks=:0.03f}\n{p_salt=:0.03f}\n{base_rate=:0.01f} sps\n{stim_rate=:0.01f} sps",
                ha="left",
                va="center",
                transform=ax[0].transAxes,
            )

        # Tidy axes
        sns.despine()
        plt.tight_layout()

        # Save plot
        is_tagged = False
        if optotag_rez is not None:
            is_tagged = this_cell["is_tagged"].values[0]
        if is_tagged:
            save_fn = save_folder.joinpath(f"tagged_clu_{clu:04.0f}.png")
            save_fn2 = save_folder.joinpath(f"tagged_clu_{clu:04.0f}.svg")
        else:
            save_fn = save_folder.joinpath(f"untagged_clu_{clu:04.0f}.png")
            save_fn2 = save_folder.joinpath(f"untagged_clu_{clu:04.0f}.svg")
        plt.savefig(save_fn, dpi=300, transparent=True)
        plt.savefig(save_fn2, transparent=True)
        plt.close("all")

    # Population plots - seperate by salt_p_stat <0.001
    f, ax = plt.subplots(figsize=(8, 8), ncols=2, sharex=True)
    tagged_clus = np.where(
        np.isin(cluster_ids, optotag_rez.query("is_tagged")["cluster_id"].values)
    )[0]
    untagged_clus = np.where(
        np.isin(cluster_ids, optotag_rez.query("~is_tagged")["cluster_id"].values)
    )[0]
    max_spikes = 250
    cc1 = ax[0].pcolormesh(
        peths.tscale,
        np.arange(untagged_clus.shape[0]),
        peths.means[untagged_clus],
        vmin=0,
        vmax=max_spikes,
        cmap=cmap,
    )
    cc2 = ax[1].pcolormesh(
        peths.tscale,
        np.arange(tagged_clus.shape[0]),
        peths.means[tagged_clus],
        vmin=0,
        vmax=max_spikes,
        cmap=cmap,
    )

    for aa in ax:
        aa.axvline(0, color="w", ls=":")
        aa.axvline(consideration_window, color="silver", ls=":")
        aa.set_ylabel("Units (unordered)")
        aa.set_xlabel("Time (s)")
        if stim_duration is not None:
            aa.axvline(stim_duration, color="w", ls=":")
    ax[0].set_title("Untagged")
    ax[1].set_title(
        f"Tagged ({method} p<{SALT_P_CUTOFF} and \nstims with  spikes>{MIN_PCT_TAGS_WITH_SPIKES:0.0f}%)"
    )
    cax1 = plt.colorbar(cc1)
    cax2 = plt.colorbar(cc2)
    cax1.set_ticks([0, 100, 200, 250])
    cax1.set_ticklabels(["0", "100", "200", ">250"])
    cax2.set_ticks([0, 100, 200, 250])
    cax2.set_ticklabels(["0", "100", "200", ">250"])
    cax1.set_label("F.R. (sp/s)")
    cax2.set_label("F.R. (sp/s)")
    plt.tight_layout()
    plt.savefig(save_folder.joinpath("population_tags.png"), dpi=300, transparent=True)


def kstest_optotag(
    spike_times, spike_clusters, cluster_ids, tags, window_time, stim_duration
):
    """
    Computes a one-sided Kolmogorov-Smirnov test to determine if the number of spikes after the stim is
    more than before the stim.
    Useful in the case that we do not expect latencies to be tightly aligned, which is the case for the ChRmine optotag

    Args:
        spike_times (1D numpy array): times in  seconds of every spike
        spike_clusters (1D numpy array): cluster assignments of every spike
        cluster_ids (1D numpy array): cluster ids to use.
        stim_times (1D numpy array): onset times of the optogenetic stimulus
        window_time (float, optional): window to compute spike rate in seconds. Defaults to 0.01.
        tags (AlfBunch): Opto stimulus data for the tags. Should have the attribute "intervals"
        stim_duration (float): Duration of stimulus in seconds.

    Returns:
        p (list): p-value of the null hypothesis that FR_pre = FR_post. Each entry corresponds to an entry in "cluster_id"
    """
    stim_times = tags.intervals[:, 0]
    pre_raster, post_raster = compute_pre_post_raster(
        spike_times,
        spike_clusters,
        cluster_ids,
        stim_times,
        window_time=window_time,
        stim_duration=stim_duration,
        bin_size=0.001,
        mask_dur=0.002,
    )

    p = []
    for ii, clu in enumerate(cluster_ids):
        pre = np.sum(pre_raster[:, ii, :], 1) / window_time
        post = np.sum(post_raster[:, ii, :], 1) / window_time
        p.append(scipy.stats.ks_2samp(pre, post, alternative="greater").pvalue)
    return p


def run_probe(probe_path, tags, consideration_window, wavelength, plot=False,no_matlab=False,overwrite=False):
    """
    Computes the optotag statistics on data from one probe.
    Loads in from the ALF format.
    Computes:
        1) SALT statistics
        2) Heuristics for number of stims with spikes, spike rates
        3) Kolmogorov-Smirnov test of significance

    Saves data to the probe path with the namespace "salt"
    Saves the parameters associated with the computations (e.g., SALT-p cutoff) as a JSON in the probe path
    Optionally saves figures for each unit

    Args:
        probe_path (pathlib Path): Path to the spike ALF data (e.g., <session>/alf/probeXX)
        tags (AlfBunch): Object of opto stimulations for just the tagging stimulations
        consideration_window (float): Time in seconds to consider for statistical comparisons
        wavelength (int): Wavelength of the stimulus in nm.
        plot (bool, optional): If true, plots figures for each unit. Defaults to False.
    """
    spikes = alfio.load_object(probe_path, "spikes")
    clusters = alfio.load_object(probe_path, "clusters")
    cluster_ids = np.arange(len(clusters.amps),dtype='int16')
    if hasattr(clusters,'optotag') or hasattr(clusters,'isTagged'):
        if overwrite:
            print('Removing existing optotag data')
            fn,parts = alfio.filter_by(probe_path,object='clusters',attribute='optotag')
            for f in fn:
                probe_path.joinpath(f).unlink()
            fn,parts = alfio.filter_by(probe_path,object='clusters',attribute='isTagged')
            for f in fn:
                probe_path.joinpath(f).unlink()
        else:
            print('Optotag data already exists. Use --overwrite to overwrite.')
            return

    # Set the query to mark as "tagged"
    if no_matlab:
        query = "p_ks<@SALT_P_CUTOFF & pct_stims_with_spikes>@MIN_PCT_TAGS_WITH_SPIKES & stim_rate/base_rate>@RATIO"
        method = "ks"
    else:
        query = "salt_p_stat<@SALT_P_CUTOFF & pct_stims_with_spikes>@MIN_PCT_TAGS_WITH_SPIKES"
        method = "salt"

    spike_times = spikes.times
    spike_clusters = spikes.clusters

    tag_duration = np.mean(np.diff(tags.intervals, 1))
    n_tags = tags.intervals.shape[0]
    tag_onsets = tags.intervals[:, 0]
    # Compute SALT data
    if not no_matlab:
        p_stat, I_stat = run_salt(
            spike_times,
            spike_clusters,
            cluster_ids,
            tag_onsets,
            window_time=consideration_window * 50,
            stim_duration=tag_duration,
            consideration_window=consideration_window,
        )
    else:
        p_stat = np.ones_like(cluster_ids)*np.nan
        I_stat = np.ones_like(cluster_ids)*np.nan

    # Compute heuristic data
    n_stims_with_spikes, base_rate, stim_rate = compute_tagging_summary(
        spike_times,
        spike_clusters,
        cluster_ids,
        tag_onsets,
        stim_duration=tag_duration,
        window_time=consideration_window,
    )

    # Compute KStest statistic
    p_ks = kstest_optotag(
        spike_times,
        spike_clusters,
        cluster_ids,
        tags,
        window_time=consideration_window,
        stim_duration=tag_duration,
    )

    # Export to a pqt
    optotag_rez = pd.DataFrame()
    optotag_rez["cluster_id"] = clusters.metrics.cluster_id
    optotag_rez.loc[cluster_ids, "salt_p_stat"] = p_stat
    optotag_rez.loc[cluster_ids, "salt_I_stat"] = I_stat
    optotag_rez.loc[cluster_ids, "n_stims_with_spikes"] = n_stims_with_spikes
    optotag_rez.loc[cluster_ids, "pct_stims_with_spikes"] = (
        n_stims_with_spikes / n_tags * 100
    )
    optotag_rez.loc[cluster_ids, "base_rate"] = base_rate
    optotag_rez.loc[cluster_ids, "stim_rate"] = stim_rate
    optotag_rez.loc[cluster_ids, "p_ks"] = p_ks
    optotag_rez["is_tagged"] = False
    optotag_rez.loc[cluster_ids,'is_tagged'] = optotag_rez.eval(query)
    optotag_rez['method'] = method
    save_fn = probe_path.joinpath(
        alfio.spec.to_alf("clusters", "optotag", namespace="cibrrig", extension="pqt")
    )
    optotag_rez.to_parquet(save_fn)
    
    # Save to the clusters alf object
    is_tagged = {"isTagged": optotag_rez["is_tagged"].values}
    alfio.save_object_npy(probe_path, is_tagged, "clusters", namespace='cibrrig')
    print(f"optotagging info saved to {save_fn}.")

    if plot:
        make_plots(
            spike_times,
            spike_clusters,
            cluster_ids,
            tags,
            save_folder=probe_path.joinpath("tag_plots"),
            optotag_rez=optotag_rez,
            wavelength=wavelength,
            consideration_window=consideration_window,
            method=method
        )

    import json

    fn_parameters = probe_path.joinpath(
        alfio.spec.to_alf("optotag", "parameters", "json", "cibrrig")
    )
    params = dict(
        SALT_P_CUTOFF=SALT_P_CUTOFF,
        MIN_PCT_TAGS_WITH_SPIKES=MIN_PCT_TAGS_WITH_SPIKES,
        consideration_window=consideration_window,
        wavelength=wavelength,
        method=method,
    )
    if wavelength == 635:
        params["RATIO"] = RATIO
    with open(probe_path.joinpath(fn_parameters), "w") as fid:
        json.dump(params, fid)


def run_session(session_path, consideration_window, plot, wavelength,no_matlab=False,overwrite=False):
    """
    Run optotagging statistics on each probe in a session.
    Reads in the laser object to get the opto stimulation parameters.
    Reads in the log file to identify which stims are tags

    Args:
        session_path (pathlib Path): Alf session path (parents should be Subjects/<mouseID>)
        consideration_window (float): Time in seconds to consider for statistical comparisons
        plot (bool, optional): If true, plots figures for each unit. Defaults to False.
        wavelength (float): Wavelength  of light used for plotting purposes
    """

    # Load opto times and logs
    log_fn = list(session_path.glob("*log*.tsv"))
    assert (
        len(log_fn) == 1
    ), f"Number of log files found was {len(log_fn)}. Should be one"
    log_fn = log_fn[0]
    laser = alfio.load_object(session_path.joinpath("alf"), "laser", short_keys=True)
    log_df = pd.read_csv(log_fn, index_col=0, sep="\t")

    # Extract only tag times
    tags = extract_tagging_from_logs(log_df, laser)

    probe_paths = list(session_path.joinpath("alf").glob("probe[0-9][0-9]"))
    for probe in probe_paths:
        run_probe(
            probe,
            tags,
            consideration_window=consideration_window,
            wavelength=wavelength,
            plot=plot,
            no_matlab=no_matlab,
            overwrite=overwrite,
        )


@click.command()
@click.argument("session_path")
@click.option(
    "-w",
    "--consideration_window",
    default=0.01,
    help="Option to change how much of the stimulus time to consider as important. Longer times may be needed for ChRmine",
)
@click.option(
    "-l",
    "--wavelength",
    default=473,
    help="set wavelength of light (changes color of plots.)",
)
@click.option("-p", "--plot", is_flag=True, help="Flag to make plots for each cell")
@click.option("--no-matlab", is_flag=True, help="Flag to skip matlab")
@click.option('--overwrite', is_flag=True, help='Flag to overwrite existing optotag data')
def main(session_path, consideration_window, plot, wavelength, no_matlab,overwrite):
    """
    CLI entry to run session

    Args:
        session_path (pathlib Path): Alf session path (parents should be Subjects/<mouseID>)
        consideration_window (float): Time in seconds to consider for statistical comparisons
        plot (bool, optional): If true, plots figures for each unit. Defaults to False.
        wavelength (float): Wavelength  of light used for plotting purposes
    """
    if no_matlab:
        print('Skipping SALT computation that requires MATLAB')
    session_path = Path(session_path)
    run_session(session_path, consideration_window, plot, wavelength,no_matlab=no_matlab,overwrite=overwrite)


if __name__ == "__main__":
    main()
