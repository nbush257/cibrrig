"""
This module applies synchronization to the ALF (Alyx Filenames) standard
for all probes within a given session path. The synchronization adjusts spike times from the IMEC clock
to the NIDAQ clock.

Key Features:

- Applies synchronization to spike times.
- Saves adjusted and original spike times.
- Provides a command-line interface for session processing.
"""

from ibllib.ephys.spikes import apply_sync
from pathlib import Path
import spikeglx
import one.alf.io as alfio
import pandas as pd
import logging
import numpy as np
import click

logging.basicConfig()
_log = logging.getLogger("convert_to_alf")
_log.setLevel(logging.INFO)


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


def run_session(session_path):
    """
    Converts all sorting in a session to the ALF standard, applying synchronization and QC metrics.

    Assumes:
    - Spike sorting data is in the structure: <session>/alf/probeXX/
    - Raw ephys data is located in: <session>/raw_ephys_data/probeXX
    - Processed ephys data is stored as: <session>/alf/probeXX/<.si>/.preprocessed
    Args:
        session_path (Path): Path to the session directory.
    """
    session_path = Path(session_path)
    ephys_path = session_path.joinpath("raw_ephys_data")
    probes_alfs = list(session_path.joinpath("alf").glob("probe[0-9][0-9]"))

    for probe_alf in probes_alfs:
        _log.info(f"Synchronizing {probe_alf.name} with nidaq clock")

        # Synchronize spikes
        _log.info("Syncronizing to NIDAQ clock")
        raw_probe = list(ephys_path.glob(probe_alf.name))
        assert len(raw_probe) == 1, (
            f"More than one path in {ephys_path} matches {probe_alf.name}"
        )
        raw_probe = raw_probe[0]
        ap_files = list(raw_probe.rglob("*ap.bin"))
        _log.info(f"Found {len(ap_files)} ap files")

        # Adjust spike times to the NIDAQ clock
        spikes = alfio.load_object(probe_alf, "spikes")
        times_adj = sync_spikes(ap_files, spikes)

        # Save adjusted and original spike times
        _log.info("Saving adjusted and old spike times")
        times_old_fn = alfio.files.spec.to_alf(
            object="spikes", attribute="times", timescale="ephysClock", extension="npy"
        )
        times_adj_fn = alfio.files.spec.to_alf(
            object="spikes", attribute="times", extension="npy"
        )
        np.save(probe_alf.joinpath(times_old_fn), spikes.times)
        np.save(probe_alf.joinpath(times_adj_fn), times_adj)


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
    run_session(session_path, sorting_name)


if __name__ == "__main__":
    main()
