"""
This module applies synchronization and converts spike sorting data to the ALF (Alyx Filenames) standard
for all probes within a given session path. The synchronization adjusts spike times from the IMEC clock
to the NIDAQ clock. This conversion is crucial for making data compatible with the IBL's ONE framework,
which uses ALF to handle data files systematically.

Key Features:
- Aggregates and saves cluster-level metrics from Kilosort.
- Applies synchronization to spike times.
- Converts spike sorting data to the ALF standard format.
- Saves adjusted and original spike times.
- Provides a command-line interface for session processing.

"""

try:
    from ibllib.ephys.sync import apply_sync
except ImportError:
    try:
        from ibllib.ephys.sync_probes import apply_sync
    except ImportError:
        from ibllib.ephys.spikes import apply_sync
try:
    from ibllib.ephys.qc import spike_sorting_metrics
except ImportError:
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
    Aggregates all cluster-level metrics  as computed from Spike interface from the Kilosort directory.

    Expands thedataframe to be the same size as max cluster_id

    Args:
        ks4_dir (Path): Directory where Kilosort results are stored (e.g., sorting output).

    Returns:
        pandas.DataFrame: A dataframe containing cluster metrics from the directory.
    """
    metrics_files = ks4_dir.glob("cluster_*.tsv")
    metrics = pd.read_csv(ks4_dir.joinpath("cluster_group.tsv"), sep="\t")
    for fn in metrics_files:
        if fn.name == "cluster_group.tsv":
            continue  # Don't reread group
        if fn.name == "cluster_info.tsv":
            continue  # Don't reread group
        # Skip potential_merges
        df = pd.read_csv(fn, sep="\t")
        if "cluster_id" not in df.columns:
            continue
        metrics = metrics.merge(df, on="cluster_id", how="left")
    metrics.set_index('cluster_id', inplace=True)
    full_index = pd.Index(range(metrics.index.min(), metrics.index.max() + 1))
    metrics_full = metrics.reindex(full_index)
    metrics_full.reset_index(inplace=True)
    metrics_full.rename(columns={'index': 'cluster_id'}, inplace=True)
    return metrics_full


def save_si_metrics(metrics, out_path):
    """
    Saves the cluster-level QC metrics in ALF format.

    Args:
        metrics (pandas.DataFrame): Cluster-level metrics dataframe.
        out_path (Path): Directory where the metrics file will be saved.
    """
    metrics_fn = alfio.spec.to_alf("clusters", "siMetrics", "pqt")
    metrics.to_parquet(out_path.joinpath(metrics_fn))


def get_ap_breaks_samps(ap_files):
    """
    Computes the cumulative sample lengths for multiple recordings. This is useful
    for handling recordings across multiple triggers.

    Example: If recording 1 has 110 samples and recording 2 has 24 samples, this
    function returns [0, 110, 134].

    Args:
        ap_files (list[Path]): List of paths to .ap.bin files.

    Returns:
        numpy.ndarray: Cumulative sample lengths across recordings.
    """
    breaks = [0]
    for fn in ap_files:
        SR = spikeglx.Reader(fn)
        breaks.append(breaks[-1] + SR.ns)
    breaks = np.array(breaks)
    return breaks


def sync_spikes(ap_files, spikes):
    """
    Synchronizes spike times by adjusting them to match the NIDAQ clock, based on
    synchronization files that align the IMEC clock to the NIDAQ clock. Spikes with
    negative times are set to zero.

    Args:
        ap_files (list[Path]): List of paths to .ap.bin files.
        spikes (AlfBunch): ALF object containing spike data.

    Returns:
        numpy.ndarray: Adjusted spike times after synchronization.
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
    """
    Converts Kilosort output to the ALF format via phylib and performs additional data processing such as
    computing shanks, adjusting amplitudes, and generating spike waveforms.

    Args:
        ks_path (Path): Path to the Kilosort directory.
        alf_path (Path): Output directory where ALF files will be saved.
        sample_rate (float): Sampling rate of the recording.
        ampfactor (float): Conversion factor from samples to voltage.
    """
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

    # save_si_metrics
    si_metrics = get_metrics_from_si(ks_path)
    save_si_metrics(si_metrics, alf_path)
    np.save(alf_path.joinpath("clusters.group.npy"),si_metrics['group'].values)

    # TODO: Confirm waveforms are properly scaled


def run_session(session_path, sorting_name="kilosort4"):
    """
    Converts all sorting in a session to the ALF standard, applying synchronization and QC metrics.

    Assumes:
    - Spike sorting data is in the structure: <session>/alf/probeXX/<sorting_name>
    - Raw ephys data is located in: <session>/raw_ephys_data/probeXX
    - Processed ephys data is stored as: <session>/alf/probeXX/<sorting_name>/recording.dat

    Args:
        session_path (Path): Path to the session directory.
        sorting_name (str, optional): Name of the sorting folder. Defaults to 'kilosort4'.
    """
    session_path = Path(session_path)
    ephys_path = session_path.joinpath("raw_ephys_data")
    probes_alfs = list(session_path.joinpath("alf").glob("probe[0-9][0-9]"))

    for probe_alf in probes_alfs:
        _log.info(f"Converting {probe_alf.name} from {sorting_name} to ALF")
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

        # Convert Kilosort output to ALF
        sr = spikeglx.Reader(ap_fn)
        sampling_rate = sr.fs
        s2v = sr.sample2volts[0]
        convert_model(ks4_dir, out_path, sampling_rate, s2v)

        # Compute QC metrics
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

        # Synchronize spikes
        _log.info("Syncronizing to NIDAQ clock")
        raw_probe = list(ephys_path.glob(probe_alf.name))
        assert (
            len(raw_probe) == 1
        ), f"More than one path in {ephys_path} matches {probe_alf.name}"
        raw_probe = raw_probe[0]
        ap_files = list(raw_probe.rglob("*ap.bin"))
        _log.info(f"Found {len(ap_files)} ap files")

        # Adjust spike times to the NIDAQ clock
        spikes = alfio.load_object(out_path, "spikes")
        times_adj = sync_spikes(ap_files, spikes)

        # Save adjusted and original spike times
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
@click.argument("sorting_name")
def main(session_path, sorting_name):
    """
    Command-line interface for converting spike sorting data to the ALF standard.

    Args:
        session_path (str): Path to the session directory.
        sorting_name (str): Name of the sorting.
    """
    run_session(session_path,sorting_name)


if __name__ == "__main__":
    main()
