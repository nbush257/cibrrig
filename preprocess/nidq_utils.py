'''
I/O functions with NIDAQ data specifically for some of our phsyiology needs
'''

import scipy.integrate
import numpy as np
import scipy.signal as sig
from scipy.ndimage import median_filter
import warnings
import spikeglx
import physiology
import logging
import re
import matplotlib.pyplot as plt
logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def get_triggers(session_path):
    """Looks through all the NIDQ files to extract the trigger

    Args:
        session_path (_type_): _description_
    """    
    ni_files = list(session_path.joinpath('raw_ephys_data').glob('*.nidq.bin'))
    trig_strings = [re.search('t\d{1,3}',x.stem).group() for x in ni_files]
    trig_strings.sort()
    return(trig_strings)

def get_trig_string(in_str):
    """Extract the trigger from a string (e.g., t0,t12) using re

    Args:
        in_str (_type_): _description_
    """    
    trig_string = re.search('t\d{1,3}',in_str).group()
    return(trig_string)

def _extract_ds_chan(SR,chan_id,ds_factor=10):
    assert(type(ds_factor) is int)
    sr = SR.fs
    sr_sub = sr / ds_factor
    dat = SR.read(nsel=slice(None,None,ds_factor),csel=chan_id,sync=False)
    return(dat,sr_sub)


def load_mmap(fn):
    SR = spikeglx.Reader(fn)
    mmap = SR.read()[0]
    return(mmap,SR.meta)


def binary_onsets(x,thresh):
    '''
    Get the onset and offset samples of a binary signal 
    :param x: signal
    :param thresh: Threshold
    :return: ons,offs
    '''
    xbool = x>thresh

    ons = np.where(np.diff(xbool.astype('int'))==1)[0]
    offs = np.where(np.diff(xbool.astype('int'))==-1)[0]
    if xbool[0]:
        offs = offs[1:]
    if xbool[-1]:
        ons = ons[:-1]
    if len(ons)!=len(offs):
        plt.plot(x)
        plt.axhline(thresh)
        raise ValueError('Onsets does not match offsets')
    return(ons,offs)


def get_tvec(dat,sr):
    tvec = np.linspace(0,len(dat)/sr,len(dat))
    return(tvec)


def get_tvec_from_fn(fn):
    '''
    Overload get tvec to work on a nidaq filename
    :param fn: Path object to a Nidaq file
    :return: tvec
    '''
    SR = spikeglx.Reader(fn)
    tvec = get_tvec_from_SR(SR)
    return(tvec)


def get_tvec_from_SR(SR):
    '''
    Extract the timevector given a memory map and a metafile
    :param mmap: A nidaq memory map object of aux data
    :param meta: Meta data as extracted from readSGLX
    :return: tvec
    '''
    sr = SR.fs
    n_samps = SR.ns
    tvec = np.linspace(0,n_samps/sr,n_samps)
    return(tvec)


def load_ds_pdiff(SR,chan_id,ds_factor=10,winsize=5,inhale_dir=-1):
    '''
    Load and downsample the pleth data
    :param mmap: mmap
    :param meta: metadata dict
    :param chan_id: Pleth channel index
    :param ds_factor: downsample factor
    :return:
            dat- downsampled pleth data
            sr_sub - new sampling rate
    '''
    dat,sr_sub = _extract_ds_chan(SR,chan_id,ds_factor)
    dat = dat*inhale_dir

    # Do not do any baseline correction on the PDIFF because it is AC.
    return(dat,sr_sub)


def load_ds_process_flowmeter(SR,chan_id,vin=9,ds_factor=10,inhale_dir=-1):
    raise NotImplementedError('Need to reimplement flow correction')
    # assert (type(ds_factor) is int)
    # idx = np.arange(0,SR.ns,ds_factor)
    # flow = SR.read(nsel=idx,csel=chan_id,sync=False)
    # sr = SR.fs
    # sr_sub = sr / ds_factor
    # winsize = 5
    # # Calibrate voltage to flow
    # flow_calibrated = data.calibrate_flowmeter(flow, vin=vin)
    # # Correct for bias flow
    # flow_calibrated_corrected = data.baseline_correct_integral(flow_calibrated,sr=sr,winsize=winsize)
    # # Make inhalation updward deflections
    # flow_calibrated_corrected = flow_calibrated_corrected * inhale_dir
    # return(flow_calibrated_corrected,sr_sub)


def load_dia_emg(SR,chan_id):
    '''
    Read the raw diaphragm emg
    :param mmap: memory ampped aux data
    :param meta: meta dict
    :param chan_id: channel index of the diaphragm
    :return:
        dat - the raw diaphramg
        sr - the smapling rate of the diaphragm recording
    '''
    ds_factor=1 #Do no downsampling here
    dat,sr = _extract_ds_chan(SR,chan_id,ds_factor)
    dat = dat-np.mean(dat)
    return(dat,sr)


def filt_int_ds_dia(x,sr,ds_factor=10,rel_height=0.95):
    '''
    Filter, integrate and downsample the diaphragm. Detect and summarize the diaphragm bursts
    Uses medfilt to smooth so it is a little slow, but it is worth it.
    :param x:
    :param sr:
    :param ds_factor:
    :return:
    '''
    assert(type(ds_factor) is int)

    #Remove the EKG artifact
    _log.info('Removing the EKG...')
    dia_filt,pulse = physiology.remove_EKG(x,sr,thresh=2)
    dia_filt[np.isnan(dia_filt)] = np.nanmedian(dia_filt)


    # Filter for high frequency signal

    sos = sig.butter(2,[300/sr/2,5000/sr/2],btype='bandpass',output='sos')
    dia_filt = sig.sosfilt(sos,dia_filt)

    # Use medfilt to get the smoothed rectified EMG
    _log.info('Smoothing the rectified trace...')

    window_length = int(0.05*np.round(sr))+1
    if window_length%2==0:
        window_length+=1
    dd = median_filter(np.abs(dia_filt),window_length)
    # Smooth it out a little more
    window_length = int(0.01*np.round(sr))+1
    if window_length%2==0:
        window_length+=1
    dia_smooth = sig.savgol_filter(dd,window_length=window_length,polyorder=1)

    # Downsample because we don't need this at the original smapling rate
    dia_sub = dia_smooth[::ds_factor]
    sr_sub = sr/ds_factor

    # get the burst statistics
    warnings.filterwarnings('ignore')
    dia_df = physiology.burst_stats_dia(dia_sub,sr_sub,rel_height=rel_height)
    warnings.filterwarnings('default')

    HR,heartbeats = physiology.get_hr_from_dia(pulse/ds_factor,dia_df,sr_sub)

    # Normalize the integrated diaphragm to a z-score.
    dia_df['amp_z'] = dia_df['amp']/np.std(dia_sub)
    dia_sub = dia_sub/np.std(dia_sub)
    _log.info('Done processing diaphragm')

    return(dia_df,dia_sub,sr_sub,HR,dia_filt,heartbeats)


def extract_hr_channel(SR,ekg_chan=2):
    '''
    If the ekg is recorded on a separate channel, extract it here
    return: bpm, pulse times
    '''
    ds_factor=1 # No downsampling here
    dat,sr = _extract_ds_chan(SR,ekg_chan,ds_factor=ds_factor) 
    dat = dat-np.mean(dat)

    heartbeats = physiology.extract_hr_from_ekg(dat,sr)

    return(heartbeats)


def extract_temp(SR,temp_chan=7,ds_factor=10):
    """
    Extract the temperature from the FHC DC temp controller. Assumes the manufacturers calibration
    :param temp_chan:
    :return:
    """
    assert(type(ds_factor) is int)
    dat,sr =_extract_ds_chan(SR,temp_chan,ds_factor=ds_factor)
    # 0v=25C, 2V = 45C, 100mv=1C
    vout_map = [0,2]
    temp_map = [25,45]
    temp_f = scipy.interpolate.interp1d(vout_map, temp_map)
    temp_out = temp_f(dat)
    temp_out = scipy.signal.savgol_filter(temp_out,101,1)
    return(temp_out)


def filt_int_ds_arbitrary(x,sr,ds_factor=10):
    assert(type(ds_factor) is int)

    # Filter for high frequency signal

    sos = sig.butter(2,[300/sr/2,5000/sr/2],btype='bandpass',output='sos')
    x_filt = sig.sosfilt(sos,x)

    # Use medfilt to get the smoothed rectified EMG
    _log.info('Smoothing the rectified trace...')

    window_length = int(0.05*np.round(sr))+1
    if window_length%2==0:
        window_length+=1
    dd = median_filter(np.abs(x_filt),window_length)
    # Smooth it out a little more
    window_length = int(0.01*np.round(sr))+1
    if window_length%2==0:
        window_length+=1
    dia_smooth = sig.savgol_filter(dd,window_length=window_length,polyorder=1)

    # Downsample because we don't need this at the original smapling rate
    x_sub = dia_smooth[::ds_factor]
    sr_sub = sr/ds_factor

    # Normalize the integrated diaphragm to a z-score.
    x_sub = x_sub/np.std(x_sub)
    _log.info('Done processing signal')
    return(x_sub,sr_sub,x_filt)
