"""
This module runs the preprocessing pipeline for a session.
It extracts sync, video frame, opto, and physiology information from the session.
(DEPRECATED runs ephysqc on the session.)
"""

import logging
import sys
from pathlib import Path
import click
import matplotlib.pyplot as plt
from cibrrig.preprocess import (
    extract_frame_times,
    extract_opto_times,
    extract_physiology,
    extract_sync_times,
)
if sys.platform == "linux":
    import matplotlib

    matplotlib.use("Agg")
import numpy as np
import one.alf.io as alfio
from brainbox.ephys_plots import image_rms_plot
from ibllib.plots.figures import remove_axis_outline, set_axis_label_size
from ibllib.ephys.ephysqc import EphysQC, extract_rmsmap

logging.basicConfig()
_log = logging.getLogger("PIPELINE")
_log.setLevel(logging.INFO)


def plot_QC(ephysQC):
    _log.info(f"Plotting QC for {ephysQC.probe_path.name}")
    pname = ephysQC.probe_path.name
    fig, axs = plt.subplots(
        2,
        2,
        gridspec_kw={"width_ratios": [0.95, 0.05]},
        figsize=(12, 9),
        constrained_layout=True,
    )
    lfp = alfio.load_object(ephysQC.probe_path, "ephysTimeRmsLF", namespace="iblqc")
    _, _, _ = image_rms_plot(
        lfp.rms,
        lfp.timestamps,
        median_subtract=False,
        band="LFP",
        clim=[-35, -45],
        ax=axs[0, 0],
        cmap="inferno",
        fig_kwargs={"figsize": (8, 6)},
        display=True,
        title="LFP RMS",
    )
    set_axis_label_size(axs[0, 0], cmap=True)
    remove_axis_outline(axs[0, 1])

    ap = alfio.load_object(ephysQC.probe_path, "ephysTimeRmsAP", namespace="iblqc")
    _, _, _ = image_rms_plot(
        ap.rms,
        ap.timestamps,
        median_subtract=False,
        band="AP",
        clim=[-35, -45],
        ax=axs[1, 0],
        cmap="inferno",
        fig_kwargs={"figsize": (8, 6)},
        display=True,
        title="AP RMS",
    )
    set_axis_label_size(axs[1, 0], cmap=True)

    ap = alfio.load_object(ephysQC.probe_path, "ephysChannels", namespace="iblqc")  # NOQA
    axs[1, 1].plot(ap.rawSpikeRates, np.arange(ap.rawSpikeRates.shape[0]), color="k")
    axs[1, 1].set_xlabel("sp/s")
    plt.suptitle(f"{pname}")

    save_path = Path(ephysQC.out_path).joinpath(f"{pname}_ephys_qc.png")

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


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
        qc.run(overwrite=False)
        extract_rmsmap(qc.data.ap, out_folder=qc.out_path, overwrite=False)
        plot_QC(qc)


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

@click.command()
@click.argument("session_path")
def runQC(session_path):
    run_ephys_qc_session(session_path)

if __name__ == "__main__":
    cli()
