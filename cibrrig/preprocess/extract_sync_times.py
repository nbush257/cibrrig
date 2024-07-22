"""
Extract digital signals from the NI and IMEC Streams
"""

# Deal with IBL overwritting
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

try:
    from .nidq_utils import get_triggers
except ImportError:
    import sys

    sys.path.append("../")
    from nidq_utils import get_triggers
logging.basicConfig()
_log = logging.getLogger("extract_sync_times")
_log.setLevel(logging.INFO)


def run(session_path, debug=False, no_display=False):
    display = not no_display
    type = None
    session_path = Path(session_path)
    ephys_path = session_path.joinpath("raw_ephys_data")

    triggers = get_triggers(session_path)
    for trig in triggers:
        ni_fn = list(ephys_path.glob(f"*{trig}.nidq*.bin"))
        assert (len(ni_fn)) == 1, "Incorrect number of NI files found"
        ni_fn = ni_fn[0]
        label = Path(ni_fn.stem).stem
        _log.info(f"Extracting sync from {ni_fn}")
        sync_nidq = _sync_to_alf(ni_fn, parts=label)
        alfio.save_object_npy(
            ni_fn.parent, sync_nidq, "sync", parts=trig, namespace="spikeglx"
        )
        sync_map = spikeglx.get_sync_map(ni_fn.parent)
        sync_nidq = get_sync_fronts(sync_nidq, sync_map["imec_sync"])

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


@click.command()
@click.argument("session_path")
@click.option("--debug", is_flag=bool, help="Sets logging level to DEBUG")
@click.option("--no_display", is_flag=bool, help="Toggles display")
def main(session_path, debug, no_display):
    run(session_path, debug, no_display)


if __name__ == "__main__":
    main()
