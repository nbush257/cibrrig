"""
Apply sync and export to alf for all probes given a session path
"""

from ibllib.ephys.spikes import ks2_to_alf, apply_sync
from ibllib.ephys.ephysqc import spike_sorting_metrics
from pathlib import Path
import spikeglx
import one.alf.io as alfio
import pandas as pd
import logging
import numpy as np
import click
from phylib.io.model import TemplateModel
import phylib.io.alf

logging.basicConfig()
_log = logging.getLogger("convert_to_alf")
_log.setLevel(logging.INFO)


def get_metrics_from_si(ks4_dir):
    """
    Aggregate all the cluster level metrics from the kilosort directory

    Args:
        ks4_dir (Pathlib Path): Kilosort directory
    """
    metrics_files = ks4_dir.glob("*.tsv")
    metrics = pd.read_csv(ks4_dir.joinpath("cluster_group.tsv"), sep="\t")
    for fn in metrics_files:
        if fn.name == "cluster_group.tsv":
            continue  # Don't reread group
        df = pd.read_csv(fn, sep="\t")
        if "cluster_id" not in df.columns:
            continue
        metrics = metrics.merge(df, on="cluster_id", how="left")
    return metrics


def save_metrics(metrics, out_path):
    """
    Convinience function to save the cluster-level QC metrics
    Args:
        metrics (pandas dataframe): cluster level QC metrics
        out_path (Pathlib Path): where to save the metrics file
    """
    metrics_fn = alfio.spec.to_alf("clusters", "metrics", "pqt")
    metrics.to_parquet(out_path.joinpath(metrics_fn))


def get_ap_breaks_samps(ap_files):
    """
    Return a numpy array of the cumulative sample length for each recording if there are multiple triggers
    Example: recording 1 has 110 samples and recording 2 has 24 samples. Returns: [0,110,134]

    Args:
        ap_files (list): list of Paths to  ap.bin files
    """
    breaks = [0]
    for fn in ap_files:
        SR = spikeglx.Reader(fn)
        breaks.append(breaks[-1] + SR.ns)
    breaks = np.array(breaks)
    return breaks


def sync_spikes(ap_files, spikes):
    """Finds the computed synchronization files that
    line up the IMEC clock to the NIDAQ clock and adjusts the spikes to the
    NIDAQ clock. Spikes that occur at negative timea re set to zero

    Args:
        ap_files (list): list of Paths to  ap.bin files
        spikes (AlfBunch): spikes alf object
    """
    breaks_samps = get_ap_breaks_samps(ap_files)
    rec_idx = np.searchsorted(breaks_samps, spikes.samples, side="right") - 1
    all_times_adj = []
    for ii, ap_file in enumerate(ap_files):
        parts = alfio.files.filename_parts(ap_file.name)
        sync_file = alfio.filter_by(ap_file.parent, object=parts[1], extra="sync")[0][0]
        sync_file = ap_file.parent.joinpath(sync_file)
        this_rec_times = spikes.times[rec_idx == ii]
        times_adj = apply_sync(sync_file, this_rec_times)
        all_times_adj.append(times_adj)
    all_times_adj = np.concatenate(all_times_adj)

    all_times_adj[all_times_adj < 0] = 0

    return all_times_adj


def convert_model(ks_path, alf_path, sample_rate, ampfactor):
    mdl = TemplateModel(dir_path=ks_path, sample_rate=sample_rate)

    ac = phylib.io.alf.EphysAlfCreator(mdl)
    ac.convert(alf_path, ampfactor=ampfactor)
    ac.make_depths()

    cluster_shanks = mdl.channel_shanks[mdl.clusters_channels]
    np.save(alf_path.joinpath("clusters.shanks.npy"), cluster_shanks)

    # Change amplitudes to properly scaled
    np.save(alf_path.joinpath("spikes.amps.npy"), np.abs(mdl.amplitudes))
    # Need to recompute since clusters have been modified
    cluster_amps = np.zeros(mdl.n_clusters) * np.nan
    for ii in mdl.cluster_ids:
        cluster_amps[ii] = np.mean(np.abs(mdl.amplitudes[mdl.spike_clusters == ii]))
    np.save(alf_path.joinpath("clusters.amps.npy"), cluster_amps)

    # TODO: Confirm waveforms are properly scaled


def run_session(session_path, sorting_name="kilosort4"):
    """Convert all sorting in the session to alf standard.
    Assumes spikesorting data lives in this structure: <session>/alf/probeXX/<sorting_name>
    Assumes the processed ephys data live in <session>/alf/probeXX/<sorting_name>/recording.dat
    Assumes raw ephys data lives in <session>/raw_ephys_data/probeXX

    Applies synchronization to the spike times. Original spike times are kept with the timescale "ephysClock"

    Args:
        session_path (pathlib Path): Path to the session
        sorting_name (str, optional): name of the sorting folder. Defaults to 'kilosort4'.
    """
    session_path = Path(session_path)
    ephys_path = session_path.joinpath("raw_ephys_data")
    probes_alfs = list(session_path.joinpath("alf").glob("probe[0-9][0-9]"))

    for probe_alf in probes_alfs:
        _log.info(f"Converting {probe_alf.name}")
        ap_fn = next(ephys_path.joinpath(probe_alf.name).rglob("*ap.bin"))

        # Get paths
        ks4_dir = probe_alf.joinpath(sorting_name)
        out_path = probe_alf

        # Extract cluster shanks
        cluster_shanks_df = pd.read_csv(
            ks4_dir.joinpath("cluster_channel_group.tsv"), sep="\t"
        )
        np.save(
            ks4_dir.joinpath("cluster_shanks.npy"),
            cluster_shanks_df["channel_group"].values,
        )

        # Convert to ALF
        # ks2_to_alf(ks4_dir,bin_path,out_path)
        sr = spikeglx.Reader(ap_fn)
        sampling_rate = sr.fs
        s2v = sr.sample2volts[0]
        convert_model(ks4_dir, out_path, sampling_rate, s2v)

        spikes = alfio.load_object(out_path, "spikes")
        cluster_ids = np.arange(spikes.clusters.max() + 1)
        df_units, drift = spike_sorting_metrics(
            spikes.times,
            spikes.clusters,
            spikes.amps,
            spikes.depths,
            cluster_ids=cluster_ids,
        )
        df_units.to_parquet(out_path.joinpath("clusters.metrics.pqt"))

        # Get ap files for synchronizing
        _log.info("Syncronizing to NIDAQ clock")
        raw_probe = list(ephys_path.glob(probe_alf.name))
        assert (
            len(raw_probe) == 1
        ), f"More than one path in {ephys_path} matches {probe_alf.name}"
        raw_probe = raw_probe[0]
        ap_files = list(raw_probe.rglob("*ap.bin"))
        _log.info(f"Found {len(ap_files)} ap files")

        # Adjust spikes from the IMEC clock to the NIDAQ clock
        spikes = alfio.load_object(out_path, "spikes")
        times_adj = sync_spikes(ap_files, spikes)

        # Output
        _log.info("Saving adjusted and old spike times")
        times_old_fn = alfio.files.spec.to_alf(
            object="spikes", attribute="times", timescale="ephysClock", extension="npy"
        )
        times_adj_fn = alfio.files.spec.to_alf(
            object="spikes", attribute="times", extension="npy"
        )
        np.save(out_path.joinpath(times_old_fn), spikes.times)
        np.save(out_path.joinpath(times_adj_fn), times_adj)


@click.command()
@click.argument("session_path")
def main(session_path):
    run_session(session_path)


if __name__ == "__main__":
    main()
