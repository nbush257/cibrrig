import click
import spikeglx
from pathlib import Path
import numpy as np
import re
import one.alf.io as alfio
import logging
from one.alf import spec

try:
    from .nidq_utils import get_trig_string
except ImportError:
    import sys

    sys.path.append("../")
    from nidq_utils import get_trig_string
logging.basicConfig()
_log = logging.getLogger("extract_camera_frames")
_log.setLevel(logging.INFO)


def _describe_framerate(frame_times):
    """
    Compute some QC on the frame times and print to log.
    Mostly to make sure that the frame rate is not weird.

    Args:
        frame_times (np.ndarray): Times of each frame in seconds.
    """
    if len(frame_times) == 0:
        _log.warning("No Frames found")
        return
    n_frames = frame_times.shape[0]
    framerate = np.mean(1 / np.diff(frame_times))
    framerate_std = np.std(1 / np.diff(frame_times))
    framerate_range = (
        np.min(1 / np.diff(frame_times)),
        np.max(1 / np.diff(frame_times)),
    )
    _log.info(f"Found {n_frames} frames")
    _log.info(f"Mean frame rate of {framerate:0.2f} fps")
    _log.info(f"S.D. frame rate of {framerate_std:0.2f} fps")
    _log.info(
        f"Framerate min:{framerate_range[0]:0.2f}fps\tmax:{framerate_range[1]:0.2f}fps"
    )


def process_rec_ni(ni_fn, trig_chan=6, verbose=True):
    """
    Extract frame times from the NI data.

    Args:
        ni_fn (Path): Path to the NI data file.
        trig_chan (int, optional): Digital channel of the frame trigger. Defaults to 6.
        verbose (bool, optional): If True, print additional information. Defaults to True.

    Returns:
        np.ndarray: Array of frame times in seconds.
    """
    SR = spikeglx.Reader(ni_fn)
    trig = SR.read_sync_digital(_slice=slice(None, None))[:, trig_chan]
    frame_samps = np.where(np.diff(trig) > 0)[0] + 1
    frame_times = frame_samps / SR.fs
    _describe_framerate(frame_times)
    return (frame_samps, frame_times)


def process_rec_extracted(ni_fn, trig_chan=6):
    """
    Process extracted NI data to get frame times.

    Args:
        ni_fn (Path): Path to the NI data file.
        trig_chan (int, optional): Digital channel of the frame trigger. Defaults to 6.

    Returns:
        np.ndarray: Array of frame times in seconds.
    """
    trig_string = get_trig_string(ni_fn.stem)
    alfname = dict(
        object="sync", namespace="spikeglx", extra=trig_string, short_keys=True
    )
    sync = alfio.load_object(ni_fn.parent, **alfname)
    idx = np.logical_and(sync.polarities == 1, sync.channels == trig_chan)
    frame_times = sync.times[idx]
    _describe_framerate(frame_times)

    if len(frame_times) == 0:
        return None

    return frame_times


def get_camera_chans(session_path):
    """
    Get the camera channels from the session path.

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
        if "camera" in k:
            chans_to_extract.append(v)
            labels.append(k)
    return (chans_to_extract, labels)


def run_session(session_path):
    """
    Run on an entire session and use the wiring JSON.

    Args:
        session_path (Path): Path to the session data.
    """
    session_path = Path(session_path)
    dest_path = session_path.joinpath("alf")
    dest_path.mkdir(exist_ok=True)

    ni_list = list(session_path.glob("raw_ephys_data/*nidq.bin"))
    ni_list.sort()
    _log.info(f"NI LIST: {ni_list}")

    #  Allow user to run as a regular command line
    chans, cams = get_camera_chans(session_path)
    if len(chans) == 0:
        _log.warning("No camera found in wiring. Skipping frame extraction")
        return

    for ni_fn in ni_list:
        _log.info(f"Processing {ni_fn}")
        trig_string = get_trig_string(ni_fn.stem)

        for chan, cam in zip(chans, cams):
            frame_times = process_rec_extracted(ni_fn, trig_chan=chan)
            if frame_times is not None:
                fn = spec.to_alf(cam, "times", "npy", "cibrrig", extra=trig_string)
                np.save(dest_path.joinpath(fn), frame_times)

        _log.info("done.")
    _log.info("Done with all!")


def run_file(ni_fn, chan, label="camera"):
    """
    Run directly on a ni data file and expected channel for frame triggers.

    Args:
        ni_fn (Path): Path to the NI data file.
        chan (int): Channel to extract.
        label (str, optional): Label for the channel. Defaults to "camera".
    """
    trig_string = re.search("t\d{1,3}", ni_fn.stem).group()

    frame_samps, frame_times = process_rec_ni(ni_fn, trig_chan=chan)
    fn = spec.to_alf(label, "times", "npy", "cibrrig", extra=trig_string)
    np.save(ni_fn.parent.joinpath(fn), frame_times)


def run(input_path, trig_chan=None, label="camera"):
    """
    Run the extraction process on the given input path.

    Args:
        input_path (str): Path to the input data (file or directory).
        trig_chan (int, optional): Digital channel of the frame trigger. Defaults to None.
        label (str, optional): Label for the channel. Defaults to "camera".
    """
    input_path = Path(input_path)
    if input_path.is_dir():
        run_session(input_path)
    elif input_path.is_file():
        if trig_chan is None:
            while True:
                try:
                    trig_chan = int(input("What channel should we extract?"))
                    break
                except Exception:
                    pass
        run_file(input_path, trig_chan, label)
    else:
        raise ValueError("Input not a valid session path or file")


@click.command()
@click.argument("input_path")
@click.option(
    "--trig_chan",
    "-c",
    default=None,
    type=int,
    help="Digital channel of the frame_trigger",
    show_default=True,
)
@click.option("-l", "--label", default="camera")
def main(input_path, trig_chan, label):
    """
    Main entry point to extract frame times from NI data.

    Args:
        input_path (str): Path to the input data (file or directory).
        trig_chan (int, optional): Digital channel of the frame trigger. Defaults to None.
        label (str, optional): Label for the channel. Defaults to "camera".
    """
    run(input_path, trig_chan, label)


if __name__ == "__main__":
    main()
