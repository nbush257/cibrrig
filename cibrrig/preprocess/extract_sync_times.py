"""
Extract digital signals from the NI and IMEC Streams
"""

import spikeglx
from pathlib import Path
import numpy as np
import one.alf.io as alfio
import click
import logging
from ibllib.io.extractors.ephys_fpga import get_sync_fronts, _sync_to_alf
from ibllib.ephys.sync_probes import (
    sync_probe_front_times,
    _save_timestamps_npy,
    _check_diff_3b,
)
import matplotlib.pyplot as plt
from cibrrig.preprocess.nidq_utils import get_triggers

logging.basicConfig()
_log = logging.getLogger("extract_sync_times")
_log.setLevel(logging.INFO)
MAP_TO_INTERVALS = [] # set of keys that should be mapped to intervals instead of staying as polarities
OMIT_SIGNALS = ['imec_sync','record'] # Signals we do not want to extract (they will alway be in the sync file)

def run(session_path, debug=False, no_display=False):
    """
    Extract synchronization times from the session data.

    Extract times, directions, and channels for all digital signals in the session data.

    Args:
        session_path (str, Path): Path to the session data.
        debug (bool, optional): If True, sets logging level to DEBUG. Defaults to False.
        no_display (bool, optional): If True, disables display of plots. Defaults to False.

    Returns:
        None
    """
    display = not no_display
    type = None
    session_path = Path(session_path)
    ephys_path = session_path.joinpath("raw_ephys_data")

    # Find all the triggers recorded in the session (i.e., gate)
    triggers = get_triggers(session_path)
    for trig in triggers:
        # Extract digital signals from the NI Stream
        ni_fn = list(ephys_path.glob(f"*{trig}.nidq*.bin"))
        assert (
            len(ni_fn)
        ) == 1, f"More than one NI file found. found {len(ni_fn)} files"
        ni_fn = ni_fn[0]
        label = Path(ni_fn.stem).stem
        _log.info(f"Extracting sync from {ni_fn}")
        sync_nidq = _sync_to_alf(ni_fn, parts=label)
        alfio.save_object_npy(
            ni_fn.parent, sync_nidq, "sync", parts=trig, namespace="spikeglx"
        )
        sync_map = spikeglx.get_sync_map(ni_fn.parent)
        sync_nidq = get_sync_fronts(sync_nidq, sync_map["imec_sync"])

        # Extract sync from the IMEC Stream for all probes
        probe_fns = list(ephys_path.rglob(f"*{trig}.imec*.ap.bin"))
        for probe_fn in probe_fns:
            _log.info(f"Extracting sync from {probe_fn}")
            md = spikeglx.read_meta_data(probe_fn.with_suffix(".meta"))
            sr = spikeglx._get_fs_from_meta(md)
            label = Path(probe_fn.stem).stem

            sync_probe = _sync_to_alf(probe_fn, parts=label)
            out_files = alfio.save_object_npy(
                probe_fn.parent, sync_nidq, "sync", parts=trig, namespace="spikeglx"
            )

            sync_map = spikeglx.get_sync_map(probe_fn.parent)
            sync_probe = get_sync_fronts(sync_probe, sync_map["imec_sync"])

            assert np.isclose(
                sync_nidq.times.size, sync_probe.times.size, rtol=0.1
            ), "Sync Fronts do not match"
            sync_idx = np.min([sync_nidq.times.size, sync_probe.times.size])

            qcdiff = _check_diff_3b(sync_probe)
            if not qcdiff:
                type_probe = type or "exact"
            else:
                type_probe = type or "smooth"
            timestamps, qc = sync_probe_front_times(
                sync_probe.times[:sync_idx],
                sync_nidq.times[:sync_idx],
                sr,
                display=display,
                type=type_probe,
                tol=2.5,
            )
            if display:
                plt.savefig(probe_fn.parent.joinpath(f"sync{label}.png"), dpi=300)
                plt.close("all")

            # Hack
            ef = alfio.Bunch()
            ef["ap"] = probe_fn
            out_files.extend(_save_timestamps_npy(ef, timestamps, sr))

    # sync2alf(session_path)

def sync2alf(session_path):
    raw_ephys_path = session_path.joinpath("raw_ephys_data")
    alf_path = session_path.joinpath("alf")
    alf_path.mkdir(exist_ok=True)
    triggers = get_triggers(session_path)
    
    sync_map = spikeglx.get_sync_map(raw_ephys_path)
    dig_sync = {k: v for k, v in sync_map.items() if v<16}

    for trig in triggers:
        ni_fn = list(raw_ephys_path.glob(f"*{trig}.nidq*.bin"))
        assert (
            len(ni_fn)
        ) == 1, f"More than one NI file found. found {len(ni_fn)} files"
        ni_fn = ni_fn[0]
        ni = spikeglx.Reader(ni_fn)        
        rec_duration = ni.ns / ni.fs
        sync = alfio.load_object(raw_ephys_path,'sync',namespace='spikeglx',extra=trig,short_keys=True)
        for label,chan in dig_sync.items():
            idx = sync.channels == chan
            if label in OMIT_SIGNALS:
                continue
            if 'laser' in label:
                continue
            if label in MAP_TO_INTERVALS:
                onsets = sync.times[idx][sync.polarities[idx] == 1]
                offsets = sync.times[idx][sync.polarities[idx] == -1]

                # Deal with digital value being high at the start or end of the recording
                if offsets[0] < onsets[0]:
                    onsets = np.insert(onsets,0,0)
                if offsets[-1] < onsets[-1]:
                    offsets = np.append(offsets,rec_duration)
                
                # If the number of onsets and offsets do not match, log a warning and save the times and polarities
                if len(onsets) != len(offsets):
                    _log.warning(f"Number of onsets and offsets do not match for {label}")
                    output  = {'times':sync.times[idx],'polarities':sync.polarities[idx]}
                else:
                    output = {'intervals':np.c_[onsets,offsets]}
            else:
                output  = {'times':sync.times[idx],'polarities':sync.polarities[idx]}
            alfio.save_object_npy(alf_path,output,label,namespace='spikeglx',parts=trig)
            
        




@click.command()
@click.argument("session_path")
@click.option("--debug", is_flag=bool, help="Sets logging level to DEBUG")
@click.option("--no_display", is_flag=bool, help="Toggles display")
def main(session_path, debug, no_display):
    """
    Script entry point to extract digital (sync) signals.

    Args:
        session_path (str): Path to the session data.
        debug (bool): If True, sets logging level to DEBUG.
        no_display (bool): If True, disables display of plots.

    Returns:
        None
    """
    run(session_path, debug, no_display)


if __name__ == "__main__":
    main()
