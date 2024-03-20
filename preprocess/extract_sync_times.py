'''
Extract digital signals from the NI and IMEC Streams
'''
#TODO: check for wiring file 
#Deal with IBL overwritting
import spikeglx
from pathlib import Path
import numpy as np
import spikeinterface.full as si
from ibllib.ephys import sync_probes
from ibllib.io.extractors import ephys_fpga
import ibldsp.utils 
import one.alf.io as alfio
import click
import logging
from one.alf import spec
import json
import re
from ibllib.io.extractors.ephys_fpga import get_sync_fronts, get_ibl_sync_map,_sync_to_alf
import one.alf.exceptions
from ibllib.ephys.sync_probes import sync_probe_front_times,_save_timestamps_npy,_check_diff_3b
import matplotlib.pyplot as plt
logging.basicConfig()
_log = logging.getLogger('extract_sync_times')
_log.setLevel(logging.INFO)


# Channels are hardcoded for the digital line using spike interface.
# We are using spikeinterface to read the raw data files because the IBL 
# Code cannot handle the commercial NP2 probes (2013) at this time
IMEC_CHAN = '#SY0'
NI_CHAN = '#XD0'

def _extract_sync(recording,segment_id):
    """DEPRECATED (Works with spikeinterface)

    Args:
        recording (_type_): _description_
        segment_id (_type_): _description_
    """    
    segment = recording.select_segments(segment_id)
    stream = recording.stream_id
    if stream == 'nidq':
        chan=NI_CHAN
    else:
        chan=IMEC_CHAN
    dig = segment.get_traces(channel_ids = [stream+chan])
    dig = spikeglx.split_sync(dig)
    ind, polarities = ibldsp.utils.fronts(dig,axis=0)
    samps,chans = ind

    sync = alfio.Bunch()
    sync['times'] = segment.sample_index_to_time(samps)
    sync['chans'] = chans
    sync['polarities'] = polarities
    return(sync)


def _get_triggers(session_path):
    # get the NI files:
    ni_files = list(session_path.joinpath('raw_ephys_data').glob('*.nidq.bin'))
    trig_strings = [re.search('t\d{1,3}',x.stem).group() for x in ni_files]
    return(trig_strings)



@click.command()
@click.argument('session_path')
@click.option('--debug',is_flag=bool,help='Sets logging level to DEBUG')
@click.option('--display',is_flag=bool,help='Toggles display')
def main(session_path,debug,display):
    type = None
    session_path = Path(session_path)
    ephys_path = session_path.joinpath('raw_ephys_data')

    triggers = _get_triggers(session_path)
    for trig in triggers:

        ni_fn = list(ephys_path.glob(f'*{trig}.nidq*.bin'))
        assert(len(ni_fn))==1,'Incorrect number of NI files found'
        ni_fn = ni_fn[0]
        label = Path(ni_fn.stem).stem
        sync_nidq= _sync_to_alf(ni_fn,save=True,parts=label)[0]
        sync_map = spikeglx.get_sync_map(ni_fn.parent)
        sync_nidq = get_sync_fronts(sync_nidq, sync_map['imec_sync'])

        probe_fns = list(ephys_path.rglob(f'*{trig}.imec*.ap.bin'))
        for probe_fn in probe_fns:
            md = spikeglx.read_meta_data(probe_fn.with_suffix('.meta'))
            sr =spikeglx._get_fs_from_meta(md)
            label = Path(probe_fn.stem).stem

            sync_probe,out_files = _sync_to_alf(probe_fn,save=True,parts=label)
            sync_map = spikeglx.get_sync_map(probe_fn.parent)
            sync_probe = get_sync_fronts(sync_probe, sync_map['imec_sync'])

            assert np.isclose(sync_nidq.times.size, sync_probe.times.size, rtol=0.1),'Sync Fronts do not match'
            sync_idx = np.min([sync_nidq.times.size, sync_probe.times.size])
            
            qcdiff = _check_diff_3b(sync_probe)
            if not qcdiff:
                qc_all = False
                type_probe = type or 'exact'
            else:
                type_probe = type or 'smooth'
            timestamps, qc = sync_probe_front_times(sync_probe.times[:sync_idx], sync_nidq.times[:sync_idx], sr,
                                                    display=display, type=type_probe, tol=2.5)
            if display:
                plt.savefig(probe_fn.parent.joinpath(f'sync{label}.png'),dpi=300)
                plt.close('all')
            
            # Hack 
            ef = alfio.Bunch()
            ef['ap'] = probe_fn
            out_files.extend(_save_timestamps_npy(ef, timestamps, sr))


if __name__ == '__main__':
    main()