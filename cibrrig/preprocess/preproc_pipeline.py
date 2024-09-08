"""
This module runs the preprocessing pipeline for a session.
It extracts sync, video frame, opto, and physiology information from the session.
(DEPRECATED runs ephysqc on the session.)
"""

import click
import logging
from ibllib.ephys.ephysqc import extract_rmsmap, EphysQC
import spikeglx
from pathlib import Path
from . import (
    extract_frame_times,
    extract_opto_times,
    extract_physiology,
    extract_sync_times,
)

logging.basicConfig()
_log = logging.getLogger("PIPELINE")
_log.setLevel(logging.INFO)


def DEPRECATED_run_ephys_qc_session(session_path):
    """
    (DEPRECATED - VERY SLOW) Get the RMS maps only on the longest recording of a session.

    Args:
        session_path (str or Path): Path to the session directory.

    Returns:
        None
    """
    efiles = spikeglx.glob_ephys_files(session_path)
    max_duration = 0
    qc_files = []
    longest_ap = None
    longest_lf = None
    for efile in efiles:
        if efile.get("ap") and efile.ap.exists():
            SR = spikeglx.Reader(efile.ap)
            if SR.ns > max_duration:
                max_duration = SR.ns
                longest_ap = efile["ap"]
                if efile.get("lf"):
                    longest_lf = efile["lf"]
    if longest_ap:
        SR = spikeglx.Reader(longest_ap)
        qc_files.extend(extract_rmsmap(SR, out_folder=None, overwrite=False))
    if longest_lf:
        SR = spikeglx.Reader(longest_lf)
        qc_files.extend(extract_rmsmap(SR, out_folder=None, overwrite=False))
    return qc_files


def run_ephys_qc_session(session_path):
    """
    Run ephysQC on all probes in the session.

    Args:
        session_path (str or Path): Path to the session directory.

    Returns:
        None
    """
    session_path = Path(session_path)
    probe_paths = list(session_path.glob("raw_ephys_data/probe[0-9][0-9]"))
    for probe_path in probe_paths:
        qc = EphysQC(
            probe_id=probe_path.name, session_path=session_path, use_alyx=False
        )
        qc.probe_path = probe_path
        qc.run()


def run(session_path, skip_ephysqc=False):
    """
    Run the preprocessing pipeline for a session.

    Args:
        session_path (str or Path): Path to the session directory.
        skip_ephysqc (bool, optional): If True, skip the ephysQC step. Defaults to False.

    Returns:
        None
    """
    _log.info("RUNNING PREPROCESSING")
    _log.info("Skipping ephysQC") if skip_ephysqc else None
    extract_sync_times.run(session_path)
    extract_frame_times.run(session_path)
    extract_opto_times.run(session_path)
    extract_physiology.run(session_path)
    if not skip_ephysqc:
        run_ephys_qc_session(session_path)


@click.command()
@click.argument("session_path", type=click.Path(exists=True))
@click.option("--skip_ephysqc", is_flag=True)
def cli(session_path, skip_ephysqc):
    """
    Command-line interface for running the preprocessing pipeline.

    Args:
        session_path (str): Path to the session directory.
        skip_ephysqc (bool): If True, skip the ephysQC step.

    Returns:
        None
    """
    run(session_path, skip_ephysqc)


if __name__ == "__main__":
    cli()
