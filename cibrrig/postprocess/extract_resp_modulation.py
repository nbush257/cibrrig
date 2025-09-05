"""
Compute respiratory related modulation for each unit
Uses a different computation than coherence
Not using coherence for the potential concerns that coherence may not be the most effective way.
"""

import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import one.alf.io as alfio

from cibrrig.analysis.singlecell import get_all_phase_curves
from cibrrig.preprocess.physiology import compute_dia_phase

import pandas as pd

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
plt.rcParams["axes.autolimit_mode"] = "round_numbers"


def get_vector_means(bins, rates):
    """
    Calculate the vector sum direction and tuning strength from a histogram of responses in polar space.
    Computes over all cells in the input rates.

    Args:
        bins (np.ndarray): Sampled bin locations from a polar histogram. Must be on [-pi, pi]
                           Assumes number of bins is the same as the number of observed rates.
                           Bin centers is a better usage.
        rates (np.ndarray): Observed rate at each bin location. Multiple units are passed in columns

    Returns:
        tuple: A tuple containing:
            - theta (np.ndarray): The vector mean direction of the input bin locations and centers.
            - L_dir (np.ndarray): The strength of the tuning as defined by Mazurek FiNC 2014. Equivalent to 1 - Circular Variance.
    """

    def _get_vector_mean(bins, rate):
        """
        Calculate the vector sum direction and tuning strength from a histogram of responses in polar space.

        Args:
            bins (np.ndarray): Sampled bin locations from a polar histogram. Assumes number of bins is the same as the number of observed rates.
                      Bin centers is a better usage.
            rate (np.ndarray): Observed rate at each bin location.

        Returns:
            tuple: A tuple containing:
            - theta (np.ndarray): The vector mean direction of the input bin locations and centers.
            - L_dir (np.ndarray): The strength of the tuning as defined by Mazurek FiNC 2014. Equivalent to 1 - Circular Variance.
        """

        # Calculate the direction tuning strength
        L_dir = np.abs(np.sum(rate * np.exp(1j * bins)) / np.sum(rate))

        # Calculate vector mean
        x = rate * np.cos(bins)
        y = rate * np.sin(bins)

        X = np.sum(x) / len(x)
        Y = np.sum(y) / len(x)

        theta = np.arctan2(Y, X)

        return theta, L_dir

    # Check if rates is 2D
    if rates.ndim != 2:
        rates = rates.reshape(-1, 1)
        _log.warning("Reshaping rates to 2D array")
    n_units = rates.shape[1]

    # Preallocate
    theta = np.full(n_units, np.nan)
    L_dir = np.full(n_units, np.nan)

    # Loop over all units
    for ii in range(n_units):
        t, L = _get_vector_mean(bins, rates[:, ii])
        theta[ii] = t
        L_dir[ii] = L
    return (theta, L_dir)


def _get_phase_max(bins, rates):
    """
    Computes the phase value of the peak of the polar curve.

    Args:
        bins (np.ndarray): Sampled bin locations from a polar histogram. Must be on [-pi, pi].
        rates (np.ndarray): Observed rate at each bin location. Multiple units are passed in columns.

    Returns:
        np.ndarray: The phase value of the peak of the polar curve.
    """
    return bins[np.argmax(rates, 0)]


def _get_phase_modulation(bins, rates):
    """
    Compute the phase modulation index as:
    \Phi = \frac{max(rates) - min(rates)}{mean(rates)}

    Args:
        bins (np.ndarray): Sampled bin locations from a polar histogram. Must be on [-pi, pi].
                           Assumes number of bins is the same as the number of observed rates.
                           Bin centers is a better usage.
        rates (np.ndarray): Observed rate at each bin location. Multiple units are passed in columns.

    Returns:
        np.ndarray: The phase modulation index for each unit.
    """
    return (np.max(rates, axis=0) - np.min(rates, 0)) / np.mean(rates, 0)


def _get_eta_squareds(rates):
    """
    Compute the eta squared for each unit in the rates matrix.

    Not currently implemented
    Args:
        rates (np.ndarray): Observed rate at each bin location. Multiple units are passed in columns

    Returns:
        np.ndarray: The eta squared value for each unit.
    """

    def _get_eta_squared(rate):
        """
        Orem and Dick 1983
        \eta^2 = \frac{\sigma_m^2}{\sigma_t^2} = \frac{\sigma_m^2}{\sigma_m^2+\sigma_2}
        """
        # This doesn't work
        raise ValueError("This doesnt work yet")
        n_bins, n_breaths = rate.shape
        df1 = n_bins - 1
        df2 = n_breaths * n_bins - n_bins
        aa = (
            n_breaths
            * np.sum((np.nanmean(rate, axis=1) - np.nanmean(rate)) ** 2)
            / (df1)
        )
        bb = np.nansum((rate - np.nanmean(rate)) ** 2) / (df2)
        from scipy.stats import f as fdist

        F = aa / bb
        p = fdist.sf(F, df1, df2)
        return p

    p = []
    n_units = rates.shape[1]
    for ii in range(n_units):
        p.append(_get_eta_squared(rates[:, ii, :]))

    return p


# TODO: If no diaphragm data, then breaths probably does not contain on_sec and off_sec...
def compute_resp_mod(
    spike_times, spike_clusters, cluster_ids, breaths, t0=None, tf=None
):
    """
    Compute respiratory modulation according to Mazurek et al. for all clusters in cluster_ids
    Implicitly computes phase from breaths.on_sec and breaths.off_sec
    Optionally, specify a window to compute respiratory modulation

    Times are in seconds

    Args:
        spike_times (np.ndarray): Array of spike times
        spike_clusters (np.ndarray): Array of cluster IDs for each spike
        cluster_ids (np.ndarray): Array of cluster IDs to compute on
        breaths (one.alf.io.AlfBunch): Breaths structure that contains on_sec and off_sec
        t0 (float): Start of the window to compute respiratory modulation. Defaults to 0.
        tf (float): End of the window to compute respiratory modulation. Defaults to the last spike or breath.

    Returns:
        np.ndarray: bins - Phase bins
        np.ndarray: rates - Spike rate as a function of phase for each cluster
        np.ndarray: sems - Spike rate standard error as a function of phase for each cluster
        np.ndarray: theta - Preferred phase for each cluster
        np.ndarray: L_dir - Respiratory modulation strength for each cluster
    """
    t0 = t0 or 0
    tf = tf or np.min(
        [np.max(spike_times), np.max(breaths.on_sec)]
    )  # Default to the last breath or spike, whichever is earlier

    _breaths = breaths.copy()
    idx = np.logical_and(spike_times > t0, spike_times < tf)
    spike_times = spike_times[idx]
    spike_clusters = spike_clusters[idx]

    idx = np.logical_and(breaths.on_sec > t0, breaths.on_sec < tf)
    _breaths.on_sec = breaths.on_sec[idx]
    _breaths.off_sec = breaths.off_sec[idx]

    bins, rates, sems, rates_raw = get_all_phase_curves(
        spike_times, spike_clusters, cluster_ids, _breaths, nbins=50
    )
    theta, L_dir = get_vector_means(bins, rates)

    return (bins, rates, sems, theta, L_dir)


# TODO: Set up to use label==1 or metrics=='good'
def run_probe(
    probe_path,
    breaths,
    t0=None,
    tf=None,
    use_good=False,
    plot_tgl=True,
    save_tgl=True,
):
    """
    Compute coherence using chronux ALF organized spike data in a probe path
    Wrapper to pass to "compute_resp_mod"

    Args:
        probe_path (Pathlib path): path to the ALF spiking data
        breaths (AlfBunch): Breaths structure that contains on_sec and off_sec
        t0 (float): Defines the start of the window to compute respiratory modulation, defaults to 0
        tf (float): Defines the end of the window to compute respiratory modulation, defaults to last spike or breath
        use_good (bool, optional): If True, only use "good" units. Defaults to False.
        plot_tgl (bool, optional): _description_. Defaults to True.
        save (bool, optional): _description_. Defaults to True.
    """

    _log.info(f"Running {probe_path.name}")
    spikes = alfio.load_object(probe_path, "spikes")
    clusters = alfio.load_object(probe_path, "clusters")
    _log.info("Loaded spikes!")
    if use_good:
        _log.warning(
            "use_good option probably does not work on all datasets as it looks for 'good' in the metrics table"
        )
        cluster_ids = clusters.metrics["cluster_id"][
            clusters.metrics.group == "good"
        ].values
        idx = np.isin(spikes.clusters, cluster_ids)
        spike_times = spikes.times[idx]
        spike_clusters = spikes.clusters[idx]
    else:
        cluster_ids = np.unique(spikes.clusters)
        spike_times = spikes.times
        spike_clusters = spikes.clusters

    if "resp_mod" in clusters.keys():
        _log.warning("Respiratory modulation already computed. Skipping")
        return 0
    _log.info("Computing...")
    bins, rates, sems, theta, L_dir = compute_resp_mod(
        spike_times, spike_clusters, cluster_ids, breaths, t0, tf
    )
    if plot_tgl:
        _log.info("Plotting...")
        sanity_check_plots(probe_path, bins, rates, sems, theta, L_dir)

    max_phase = _get_phase_max(bins, rates)

    if save_tgl:
        _log.info("Saving...")
        np.save(probe_path.joinpath("_cibrrig_clusters.respMod.npy"), L_dir)
        np.save(probe_path.joinpath("_cibrrig_clusters.preferredPhase.npy"), theta)
        np.save(
            probe_path.joinpath("_cibrrig_clusters.maxFiringRatePhase.npy"), max_phase
        )


def sanity_check_plots(probe_path, bins, rates, sems, theta, L_dir):
    """
    Make a few plots that show the respiratory modulation of individual units and the population.

    Args:
        probe_path (Path): Path to save the plots.
        bins (np.ndarray): Phase bins.
        rates (np.ndarray): Spike rate as a function of phase for each cluster.
        sems (np.ndarray): Spike rate standard error as a function of phase for each cluster.
        theta (np.ndarray): Preferred phase for each cluster.
        L_dir (np.ndarray): Respiratory modulation strength for each cluster.
    """
    f = plt.figure()
    gs = f.add_gridspec(nrows=2, ncols=5)
    pcen = np.nanpercentile(L_dir, [0, 25, 50, 75, 99])
    _L_dir = L_dir.copy()
    _L_dir[np.isnan(_L_dir)] = 0
    for ii in range(5):
        i_near = abs(_L_dir - pcen[ii]).argmin()

        ax = f.add_subplot(gs[:1, ii], projection="polar")
        ax.plot(bins, rates[:, i_near], color="k", lw=1)
        ax.fill_between(
            bins,
            rates[:, i_near] - sems[:, i_near],
            rates[:, i_near] + sems[:, i_near],
            alpha=0.3,
            color="k",
            lw=0,
        )
        ax.set_yticks(ax.get_ylim())
        ax.set_yticklabels(["", f"{ax.get_ylim()[1]:0.0f}"])

        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["", "", "", ""])
        ax.vlines(
            theta[i_near], 0, np.max(rates[:, i_near]) * L_dir[i_near], color="tab:red"
        )
        tt = bins[np.argmax(rates[:, i_near])]
        rr = np.max(rates[:, i_near])
        ax.plot(tt, rr, "o", color="tab:blue", lw=0.5, markerfacecolor="w")

        ax.set_title(f"Mod:{L_dir[i_near]:0.2f}; Phi:{theta[i_near]:0.1f}", fontsize=6)

    ax = f.add_subplot(gs[1:, :2], projection="polar")
    scatter = ax.scatter(
        theta,
        L_dir,
        c=L_dir,
        s=L_dir * 20,
        cmap="winter",
        edgecolor="w",
        linewidths=0.25,
    )
    scatter.set_clim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_ylim([0, 1.1])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["", "", "", ""])
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_label("L_dir")
    ax.set_title("Modulation by phase")

    ax = f.add_subplot(gs[1:, 3:])
    ax.hist(L_dir, bins=25, color="k", histtype="step")
    ax.hist(L_dir, bins=25, color="silver")

    ax.set_ylabel("# Units")
    ax.set_xlabel("Resp. Modulation")
    ax.set_xlim(0, 1.01)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title("Modulation histogram")
    plt.suptitle("Respiratory modulation sanity check")

    plt.savefig(probe_path.joinpath("respMod_sanity.png"), dpi=300, transparent=True)

    df_full = pd.DataFrame()
    df_full["L_dir"] = L_dir
    df_full["theta"] = theta
    lb = [0, 0.25, 0.5, 0.75]
    ub = [0.25, 0.5, 0.75, 1]
    f, ax = plt.subplots(ncols=4)

    for ii, (l, u) in enumerate(zip(lb, ub)):  # NOQA
        df = df_full.query("L_dir>@l & L_dir<@u")
        idx = df.sort_values(["theta"]).index.values
        _ax = ax[ii]
        _ax.pcolormesh(
            bins, np.arange(df.shape[0]), rates[:, idx].T / np.mean(rates, 1)
        )
        _ax.set_xlim([-np.pi, np.pi])
        _ax.set_ylim([0, df.shape[0]])
        _ax.axvline(0, color="w")
        _ax.set_xlabel("Phase ($\phi$)")
        _ax.set_title(f"Mod = [{l:0.2f},{u:0.2f}]")
    ax[0].set_ylabel("Unit")
    plt.tight_layout()
    plt.savefig(probe_path.joinpath("respMod_heatmap.png"))


def run_session(session_path, t0=None, tf=None, use_good=False, plot_tgl=True):
    """
    Run respiratory modulation computation on all probes for a session.

    Args:
        session_path (Path): Path to the session data.
        t0 (float, optional): Start of the epoch to compute on. Defaults to None.
        tf (float, optional): End of the epoch to compute on. Defaults to None.
        use_good (bool, optional): If True, only compute for the "good" units. Defaults to False.
        plot_tgl (bool, optional): If True, generate plots. Defaults to True.
    """
    _log.info(
        f"\nComputing respiratory modulation for {session_path}.\n\t{t0=}\n\t{tf=}\n\t{use_good=}"
    )
    if not alfio.exists(session_path.joinpath("alf"), "breaths"):
        _log.error(f"No extracted breaths data found at {session_path.joinpath('alf')}. Skipping session")
        return -1
    breaths = alfio.load_object(session_path.joinpath("alf"), "breaths")
    xt, x = compute_dia_phase(breaths.on_sec, breaths.off_sec)
    probe_paths = list(session_path.joinpath("alf").glob("probe[0-9][0-9]"))
    for probe in probe_paths:
        run_probe(
            probe,
            breaths,
            t0=t0,
            tf=tf,
            use_good=use_good,
            plot_tgl=plot_tgl,
            save_tgl=True,
        )
    return 0


@click.command()
@click.argument("session_path")
@click.option("--t0", default=None, type=float)
@click.option("--tf", default=None, type=float)
@click.option("--use_good", is_flag=True)
@click.option("--no_plot", is_flag=True)
def main(session_path, t0, tf, use_good, no_plot):
    session_path = Path(session_path)

    run_session(
        session_path,
        t0=t0,
        tf=tf,
        use_good=use_good,
        plot_tgl=~no_plot,
    )


if __name__ == "__main__":
    main()
