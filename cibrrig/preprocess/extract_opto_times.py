"""
This script extracts the optogenetic stimulation times from the NIDAQ file. 
It is similar to the extract_frame_times.py script, but it extracts optogenetic stimulation times instead of frame times. 
The script reads the raw data from the NIDAQ file, processes it to find the optogenetic stimulation times, and saves the results in an ALF file. 
The script can be run from the command line using the main function, which takes the input path to the NIDAQ file as an argument. 
The script also provides options to specify the optogenetic channel, voltage threshold, and label for the extracted data.
"""

import click
import spikeglx
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import json
from scipy.interpolate import interp1d

try:
    from .nidq_utils import binary_onsets, get_trig_string
except ImportError:
    import sys

    sys.path.append("../")
    from nidq_utils import get_trig_string, binary_onsets
from one.alf import spec

logging.basicConfig()
_log = logging.getLogger("extract_opto")
_log.setLevel(logging.INFO)


# TODO: add option to extract from digital


def get_opto_df(raw_opto, v_thresh, ni_sr, min_dur=0.001, max_dur=20):
    """
    Extract optogenetic stimulation events from raw opto data.

    Args:
        raw_opto (np.ndarray): Raw current sent to the laser or LED (1V/A).
        v_thresh (float): Voltage threshold to find crossing.
        ni_sr (float): Sample rate in Hz.
        min_dur (float, optional): Minimum stimulation duration in seconds. Defaults to 0.001.
        max_dur (float, optional): Maximum stimulation duration in seconds. Defaults to 20.

    Returns:
        pd.DataFrame: DataFrame with columns 'on', 'off', 'durs', 'amps', 'on_sec', 'off_sec', and 'dur_sec'.
    """
    ons, offs = binary_onsets(raw_opto, v_thresh)
    durs = offs - ons
    opto_df = pd.DataFrame()
    opto_df["on"] = ons
    opto_df["off"] = offs
    opto_df["durs"] = durs

    min_samp = ni_sr * min_dur #NOQA
    max_samp = ni_sr * max_dur #NOQA
    opto_df = opto_df.query("durs<=@max_samp & durs>=@min_samp").reset_index(drop=True)

    amps = np.zeros(opto_df.shape[0])
    for k, v in opto_df.iterrows():
        amps[k] = np.median(raw_opto[v["on"] : v["off"]])
    opto_df["amps"] = np.round(amps, 2)

    opto_df["on_sec"] = opto_df["on"] / ni_sr
    opto_df["off_sec"] = opto_df["off"] / ni_sr
    opto_df["dur_sec"] = np.round(opto_df["durs"] / ni_sr, 3)

    return opto_df


def process_rec(SR, opto_chan=5, v_thresh=0.5, **kwargs):
    """
    Create a DataFrame where each row is an opto pulse.
    Information about the pulse timing, amplitude, and duration are created.

    Args:
        SR (spikeglx.Reader): SpikeGLX reader object for the recording.
        opto_chan (int, optional): Channel number for the opto signal. Defaults to 5.
        v_thresh (float, optional): Voltage threshold to find crossing. Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with columns 'on_sec', 'off_sec', 'dur_sec', and 'amps'.
    """

    _log.info("Reading raw data...")
    raw_opto = SR.read(nsel=slice(None, None, None), csel=opto_chan)[0]
    _log.info("done")
    df = get_opto_df(raw_opto, v_thresh, SR.fs, **kwargs)
    df = df.drop(["on", "off", "durs"], axis=1)

    return df


def load_opto_calibration(session_path):
    """If an opto calibration JSON exists, load it and create a
    linear interpolation

    Args:
        session_path (_type_): _description_
    """
    calib_fn = session_path.joinpath("opto_calibration.json")
    f = _load_opto_calibration_fn(calib_fn)
    return f


def _load_opto_calibration_fn(calib_fn):
    """
    Load the opto calibration function from a file.
    If a calibration file exists, load it and create a linear interpolation function to convert command voltage to light power.

    Args:
        calib_fn (Path): Path to the calibration file.

    Returns:
        function: Interpolation function for the calibration.
    """
    if calib_fn.exists():
        with open(calib_fn, "r") as fid:
            calib = json.load(fid)
    else:
        return None
    _log.info("Computing calibration")
    x = calib["command_voltage"]
    y = calib["light_power"]
    f = interp1d(x, y)
    return f


def get_laser_chans(session_path):
    """
    Get the laser channels from the session path. 
    Looks for the sync_map and finds the channels that are labeled as laser.

    Args:
        session_path (Path): Path to the session data.

    Returns:
        tuple: A tuple containing:
            - list: Channels to extract.
            - list: Labels for the channels.
    """
    sync_map = spikeglx.get_sync_map(session_path.joinpath("raw_ephys_data"))
    chans_to_extract = []
    labels = []
    for k, v in sync_map.items():
        if "laser" in k:
            chans_to_extract.append(v)
            labels.append(k)
    return (chans_to_extract, labels)


def run_file(ni_fn, opto_chan, v_thresh, calib_fn, label="opto"):
    """
    Run the opto extraction process on a single NIDQ.bin file.

    Args:
        input_path (Path): Path to the input data file.
        opto_chan (int): Channel number for the opto signal.
        v_thresh (float): Voltage threshold to find crossing.
        calib (Path): Path to the calibration file.
        label (str): Label for the channel.
    """
    trig_string = get_trig_string(ni_fn.stem)
    calib_fcn = _load_opto_calibration_fn(calib_fn)
    SR_ni = spikeglx.Reader(ni_fn)
    df = process_rec(SR_ni, opto_chan=opto_chan, v_thresh=v_thresh)
    if calib_fcn is not None:
        df["milliwattage"] = calib_fcn(df["amps"])
    fn = spec.to_alf(label, "table", "pqt", "cibrrig", extra=trig_string)
    df.to_parquet(ni_fn.parent.joinpath(fn))


def run_session(session_path, v_thresh):
    """
    Run the opto extraction process on an entire session.

    Args:
        session_path (Path): Path to the session data.
        v_thresh (float): Voltage threshold above which a stimulation is registered.
    """
    dest_path = session_path.joinpath("alf")
    dest_path.mkdir(exist_ok=True)
    ni_list = list(session_path.rglob("*nidq.bin"))
    ni_list.sort()

    # Extract calibration
    calib_fcn = load_opto_calibration(session_path)
    chans, labels = get_laser_chans(session_path)

    _log.debug("Extracting from\n\t" + "\n\t".join([str(x) for x in ni_list]))
    for ni_fn in ni_list:
        ni_prefix = Path(ni_fn.stem).stem
        _log.info(f"Processing ni: {ni_prefix}")
        trig_string = get_trig_string(ni_fn.stem)
        SR_ni = spikeglx.Reader(ni_fn)
        for chan, label in zip(chans, labels):
            _log.debug("Assumes the laser channel is on an analog channel")
            opto_chan = (
                chan - 16
            )  # Magic number 16 because analog channel 0 maps to sync channel 16
            df = process_rec(SR_ni, opto_chan=opto_chan, v_thresh=v_thresh)

            # Calibrate if a calibration function is provided
            if calib_fcn is not None:
                df["milliwattage"] = calib_fcn(df["amps"])
                fn_amps = spec.to_alf(
                    label, "amplitudesMilliwatts", "npy", "cibrrig", extra=trig_string
                )
                amps = df["milliwattage"].values
                np.save(dest_path.joinpath(fn_amps), amps)

            # Save the data
            fn_intervals = spec.to_alf(
                label, "intervals", "npy", "cibrrig", extra=trig_string
            )
            intervals = df[["on_sec", "off_sec"]].values
            np.save(dest_path.joinpath(fn_intervals), intervals)

            fn_amps = spec.to_alf(
                label, "amplitudesVolts", "npy", "cibrrig", extra=trig_string
            )
            amps = df["amps"].values
            np.save(dest_path.joinpath(fn_amps), amps)

            # fn = spec.to_alf(label,'table','pqt','cibrrig',extra=trig_string)
            # df.to_parquet(dest_path.joinpath(fn))


def run(input_path, opto_chan=None, v_thresh=0.5, label="laser", calib=None):
    """
    Run the opto extraction process on the given input path.

    If input path is a session directory, the script will run on all the NIDQ files in the session.
    If input path is a single file, the script will run on that file.

    Args:
        input_path (str): Path to the input data (file or directory).
        opto_chan (int, optional): Channel number for the opto signal. Defaults to None.
        v_thresh (float, optional): Voltage threshold to find crossing. Defaults to 0.5.
        label (str, optional): Label for the channel. Defaults to "laser".
        calib (str, optional): Path to the calibration file. Defaults to None.
    """
    calib = Path(calib) if calib is not None else None
    input_path = Path(input_path)
    if input_path.is_dir():
        run_session(input_path, v_thresh)
    elif input_path.is_file():
        if opto_chan is None:
            while True:
                try:
                    opto_chan = int(input("What channel should we extract?"))
                    break
                except Exception:
                    pass
        run_file(input_path, opto_chan, v_thresh, calib, label)


@click.command()
@click.argument("input_path")
@click.option(
    "--opto_chan",
    "-o",
    default=None,
    type=int,
    help="Analog channel of the optogenetic pulses",
    show_default=True,
)
@click.option(
    "--v_thresh",
    "-v",
    default=0.5,
    type=float,
    help="voltage threshold to register a pulse",
    show_default=True,
)
@click.option("-l", "--label", default="laser")
@click.option("--calib", default=None)
def main(input_path, opto_chan, v_thresh, label, calib):
    """
    Main entry point for the script.

    Args:
        input_path (str): Path to the input data (file or directory).
        opto_chan (int, optional): Channel number for the opto signal. Defaults to None.
        v_thresh (float, optional): Voltage threshold to find crossing. Defaults to 0.5.
        label (str, optional): Label for the channel. Defaults to "laser".
        calib (str, optional): Path to the calibration file. Defaults to None.
    """
    run(input_path, opto_chan, v_thresh, label, calib)


if __name__ == "__main__":
    main()
