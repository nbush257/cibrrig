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
logging.basicConfig()
_log = logging.getLogger('extract_sync_times')
_log.setLevel(logging.INFO)


# Channels are hardcoded for the digital line using spike interface.
# We are using spikeinterface to read the raw data files because the IBL 
# Code cannot handle the commercial NP2 probes (2013) at this time
IMEC_CHAN = '#SY0'
NI_CHAN = '#XD0'
def _extract_sync(rec,stream,chan):
    dig = rec.get_traces(channel_ids = [stream+chan])
    dig = spikeglx.split_sync(dig)
    ind, polarities = ibldsp.utils.fronts(dig,axis=0)
    samps,chans = ind
    sync = {'times': samps/rec.sampling_frequency,
            'channels': chans,
            'polarities': polarities}
    return(sync)



@click.command()
@click.argument('session_path')
@click.option('--debug',is_flag=bool,help='Sets logging level to DEBUG')
@click.option('--display',is_flag=bool,help='Toggles display')
def main(session_path,debug,display):
    if debug:
        _log.setLevel(logging.DEBUG)

    ephys_files = spikeglx.glob_ephys_files(session_path)
    for efi in ephys_files:
        if 'nidq' in efi.keys():
            stream = 'nidq'
            chan=NI_CHAN
        else:
            stream = si.get_neo_streams('spikeglx',efi['path'])[0][0]
            chan=IMEC_CHAN
        _log.info(f'Working on {efi.label}. Stream:{stream} Chan:{chan}')

        rec = si.read_spikeglx(efi['path'], stream_name=stream, load_sync_channel=True)
        sync = _extract_sync(rec,stream,chan)
        out_files = alfio.save_object_npy(efi['path'], sync, 'sync',
                                namespace='spikeglx', parts=efi['label'])
        
        for x in out_files:
            _log.debug(f'Saved \t{str(x)}')

    sync_probes.sync(session_path,display=display)


if __name__ == '__main__':
    main()