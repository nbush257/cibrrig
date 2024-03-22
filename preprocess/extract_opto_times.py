'''
Run this script to extract onsets,offsets, durations, and amplitudes (in V) of the analog opto trace
returns a csv dataframe

# This is pretty hacky in the organization with the rest of the modules. NEB should refactor a lot of this code to work nicer with other software
# Runs in the iblenv environment
'''

import click
import sys
import spikeglx
from pathlib import Path
import numpy as np
import re
import pandas as pd  
import ibldsp.utils 
import logging
import one.alf.io as alfio
import json
from scipy.interpolate import interp1d
from nidq_utils import binary_onsets
from one.alf import spec
logging.basicConfig()
_log = logging.getLogger('extract_opto')
_log.setLevel(logging.INFO)


# TODO: add option to extract from digital 

def get_opto_df(raw_opto,v_thresh,ni_sr,min_dur=0.001,max_dur=20):
    '''
    :param raw_opto: raw current sent to the laser or LED (1V/A)
    :param v_thresh: voltage threshold to find crossing
    :param ni_sr: sample rate (Hz)
    :param min_dur: minimum stim duration in seconds
    :param max_dur: maximum stim duration in seconds
    :return: opto-df a dataframe with on, off, and amplitude
    '''
    ons,offs = binary_onsets(raw_opto,v_thresh)
    durs = offs-ons
    opto_df = pd.DataFrame()
    opto_df['on'] = ons
    opto_df['off'] = offs
    opto_df['durs'] = durs

    min_samp = ni_sr*min_dur
    max_samp = ni_sr*max_dur
    opto_df = opto_df.query('durs<=@max_samp & durs>=@min_samp').reset_index(drop=True)

    amps = np.zeros(opto_df.shape[0])
    for k,v in opto_df.iterrows():
        amps[k] = np.median(raw_opto[v['on']:v['off']])
    opto_df['amps'] = np.round(amps,2)

    opto_df['on_sec'] = opto_df['on']/ni_sr
    opto_df['off_sec'] = opto_df['off']/ni_sr
    opto_df['dur_sec'] = np.round(opto_df['durs']/ni_sr,3)

    return(opto_df)

def process_rec(SR,opto_chan=5,v_thresh=0.5,**kwargs):
    '''
    Create a dataframe where each row is an opto pulse. 
    Information about the pulse timing, amplitude, and duration are created.
    '''

    _log.info('Reading raw data...')
    raw_opto = SR.read(nsel=slice(None,None,None),csel=opto_chan)[0]
    _log.info('done')
    df = get_opto_df(raw_opto,v_thresh,SR.fs,**kwargs)
    df = df.drop(['on','off','durs'],axis=1)

    return(df)

def load_opto_calibration(session_path):
    """If an opto calibration JSON exists, load it and create a 
    linear interpolation

    Args:
        session_path (_type_): _description_
    """    
    calib_fn = session_path.joinpath('opto_calibration.json')
    f = _load_opto_calibration_fn(calib_fn)
    return(f)

def _load_opto_calibration_fn(calib_fn):
    if calib_fn.exists():
        with open(calib_fn, 'r') as fid:
            calib = json.load(fid)
    else:
        return(None)
    _log.info('Computing calibration')
    x = calib['command_voltage']
    y = calib['light_power']
    f = interp1d(x,y)
    return(f)


def get_laser_chans(session_path):
    sync_map = spikeglx.get_sync_map(session_path.joinpath('raw_ephys_data'))
    chans_to_extract = []
    labels = []
    for k,v in sync_map.items():
        if 'laser' in k:
            chans_to_extract.append(v)
            labels.append(k)
    return(chans_to_extract,labels)



def run_file(ni_fn,opto_chan,v_thresh,calib_fn,label='opto'):
    """run directly on a file and channel.

    Args:
        ni_fn (_type_): nidaq bin file
        chan (_type_): channel to extract
    """    
    trig_string = re.search('t\d{1,3}',ni_fn.stem).group()
    calib_fcn = _load_opto_calibration_fn(calib_fn)
    SR_ni = spikeglx.Reader(ni_fn)
    df = process_rec(SR_ni,opto_chan =opto_chan,v_thresh=v_thresh)
    if calib_fcn is not None:
        df['milliwattage'] = calib_fcn(df['amps'])
    fn = spec.to_alf(label,'table','pqt','cibrrig',extra=trig_string)
    df.to_parquet(ni_fn.parent.joinpath(fn))


def run_session(session_path,v_thresh):
    dest_path = session_path.joinpath('alf')
    dest_path.mkdir(exist_ok=True)
    ni_list = list(session_path.rglob('*nidq.bin'))

    # Extract calibration
    calib_fcn = load_opto_calibration(session_path)
    chans,labels = get_laser_chans(session_path)

    _log.debug('Extracting from\n\t'+'\n\t'.join([str(x) for x in ni_list]))
    for ni_fn in ni_list:
        ni_prefix = Path(ni_fn.stem).stem
        _log.info(f'Processing ni: {ni_prefix}')
        trig_string = re.search('t\d{1,3}',ni_fn.stem).group()
        SR_ni = spikeglx.Reader(ni_fn)
        for chan,label in zip(chans,labels):
            _log.debug('Assumes the laser channel is on an analog channel')
            opto_chan =chan-16 #Magic number 16 because analog channel 0 maps to sync channel 16 
            df = process_rec(SR_ni,opto_chan=opto_chan,v_thresh=v_thresh)

            if calib_fcn is not None:
                df['milliwattage'] = calib_fcn(df['amps'])
                fn_amps = spec.to_alf(label,'amplitudesMilliwatts','npy','cibrrig',extra=trig_string)
                amps = df['milliwattage'].values
                np.save(dest_path.joinpath(fn_amps),amps)

            fn_intervals = spec.to_alf(label,'intervals','npy','cibrrig',extra=trig_string)
            intervals = df[['on_sec','off_sec']].values
            np.save(dest_path.joinpath(fn_intervals),intervals)

            fn_amps = spec.to_alf(label,'amplitudesVolts','npy','cibrrig',extra=trig_string)
            amps = df['amps'].values
            np.save(dest_path.joinpath(fn_amps),amps)
            

            # fn = spec.to_alf(label,'table','pqt','cibrrig',extra=trig_string)
            # df.to_parquet(dest_path.joinpath(fn))


@click.command()
@click.argument('input_path')
@click.option('--opto_chan','-o',default=None,type=int,help = 'Analog channel of the optogenetic pulses',show_default=True)
@click.option('--v_thresh','-v',default=0.6,type=float,help = 'voltage threshold to register a pulse',show_default=True)
@click.option('-l','--label',default='laser')
@click.option('--calib',default=None)
def main(input_path,opto_chan,v_thresh,label,calib):
    calib = Path(calib) if calib is not None else None
    input_path = Path(input_path)
    if input_path.is_dir():
        run_session(input_path,v_thresh)
    elif input_path.is_file():
        if opto_chan is None:
            while True:
                try:
                    opto_chan = int(input('What channel should we extract?'))
                    break
                except:
                    pass
        run_file(input_path,opto_chan,v_thresh,calib,label)
        

if __name__ == '__main__':
    main()




