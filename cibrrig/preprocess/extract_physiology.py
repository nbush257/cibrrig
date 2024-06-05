'''
Command line script to extract and downsample physiological data from NIDAQ recording.
The recorded auxiliary data (e.g. diaphragm and pleth) is sampled at 10K.
While the high sampling rate is critical to acquire good EMG, it is excessive for both
the integrated and the pleth.

This script does the following:
1) Downsamples the Flow/Pdiff  traces by 10x to be input into breathmetrics (BM does filtering and processing)
2) Filters the EMG (300-5K)
3) Integrates the EMG with a triangular window
4) Downsamples the integrated EMG by 10x to match the pleth
5) Extracts features from the integrated EMG.
6) Extracts heart rate from EKG channel if recorded
6) Saves to alf standardized files in the session path
'''
import subprocess
import numpy as np
from pathlib import Path
import click
import pandas as pd
import spikeglx
import sys
from one.alf import spec
try:
    from . import physiology
    from . import nidq_utils
except:
    sys.path.append('../')
    import physiology
    import nidq_utils
import logging
logging.basicConfig()
_log = logging.getLogger('extract_physiology')
_log.setLevel(logging.INFO)

# This is fragile
if sys.platform == 'linux':
    BM_PATH = '/active/ramirez_j/ramirezlab/nbush/projects/cibrrig/cibrrig/preprocess'
else:
    BM_PATH = r'Y:/projects/cibrrig/cibrrig/preprocess'

def _crop_traces(t,x):
    '''
    Utility to crop both the time trace and an arbitrary vector to the same length(shortest lenght)
    useful to preven off by one errors
    :param t:
    :param x:
    :return:
    '''
    tlen = np.min([len(x), len(t)])
    return(t[:tlen],x[:tlen])


def _get_trigger_from_SR(SR):
    """Get the trigger number from the filename

    Args:
        SR (spikeglx.Reader):Spikeglx reader object  
    """    
    assert SR.type=='nidq', 'Not a nidaq bin file'
    label = SR.file_bin.with_suffix('').with_suffix('').stem
    trigger_label = label[-2:]
    _log.debug(f"Extracted trigger label is '{trigger_label}'")
    return(trigger_label)


def label_sighs(dia_df):
    is_sigh = physiology.compute_sighs(dia_df['on_sec'].values,dia_df['auc'].values)
    dia_df['breath_type'] = 'eupnea'
    dia_df.loc[is_sigh,'breath_type'] = 'sigh'
    dia_df['is_sigh'] = is_sigh
    dia_df['is_eupnea'] = dia_df.eval('~is_sigh')
    return dia_df


def run_one(SR, wiring, v_in, inhale_pos, save_path):
    DS_FACTOR=10
    # INIT output variables
    pdiff = np.array([])
    flow = np.array([])
    dia_filt= np.array([])
    dia_sub = np.array([])
    temperature = np.array([])
    heartbeats = np.array([])
    hr_bpm = np.array([])
    has_dia = False
    has_ekg = False
    has_pdiff = False
    has_temperature = False
    has_flowmeter = False

    analog_offset = 16 # Hardcoded offset for the sync map.
    dia_chan = wiring.get('diaphragm',False)
    ekg_chan = wiring.get('ekg',False)
    pdiff_chan = wiring.get('pdiff',False)
    flowmeter_chan = wiring.get('flowmeter',False)
    temp_chan = wiring.get('temperature',False)

    # Need explicit booleans because channel can be 0
    if dia_chan:
        dia_chan -= analog_offset
        has_dia = True
    if ekg_chan:
        ekg_chan -= analog_offset
        has_ekg = True
    if pdiff_chan:
        pdiff_chan -= analog_offset
        has_pdiff = True
    if flowmeter_chan:
        flowmeter_chan -= analog_offset
        has_flowmeter = True
    if temp_chan:
        temp_chan -=  analog_offset 
        has_temperature = True

    if inhale_pos:
        inhale_dir=1
    else:
        inhale_dir=-1


    save_path = save_path or SR.file_bin.parent

    # LOAD Memory map from SGLX
    sr = SR.fs
    # Get tvec
    t = nidq_utils.get_tvec_from_SR(SR)
    t_full = t.copy()
    t = t[::DS_FACTOR]
    sr_sub = sr/DS_FACTOR

    _log.info(f'Sampling rate is {sr}')
    _log.info(f'Downsampling to {sr_sub}')

    # Process diaphragm
    # Must do before explicit ekg processing because it attempts
    # to find the heartbeats, but it is not as good as the
    # explicit EKG channel and so we want to overwrite heartbeats with those data if they exist

    # Process EKG
    heartbeats = None # Initialize heartbeats to be none in the case that EKG is not recorded
    if has_ekg:
        _log.info('Processing EKG')
        heartbeats = nidq_utils.extract_hr_channel(SR,ekg_chan)
        _,hr_bpm = physiology.compute_avg_hr(heartbeats,t_target=t)

    # Process dia
    if has_dia:
        _log.info('Processing diaphragm')
        raw_dia,sr_dia = nidq_utils.load_dia_emg(SR,dia_chan)
        dia_df,dia_sub,sr_dia_sub,HR,dia_filt,heartbeats = nidq_utils.filt_int_ds_dia(raw_dia,sr_dia,ds_factor=DS_FACTOR,heartbeats=heartbeats)
        t,dia_sub = _crop_traces(t,dia_sub)


    # Process PDIFF
    if has_pdiff:
        _log.info('Processing pressure differential sensor')
        pdiff,sr_pdiff = nidq_utils.load_ds_pdiff(SR, pdiff_chan,ds_factor=DS_FACTOR,inhale_dir=inhale_dir)
        t,pdiff = _crop_traces(t,pdiff)

    # Process Flowmeter
    if has_flowmeter:
        _log.info('Processing flowmeter')
        flow,sr_flow = nidq_utils.load_ds_process_flowmeter(SR,flowmeter_chan,v_in,ds_factor=DS_FACTOR,inhale_dir=inhale_dir)
        t,flow = _crop_traces(t,flow)

    # Process Temperature
    if has_temperature:
        _log.info('Processing temperature')
        temperature = nidq_utils.extract_temp(SR,temp_chan,ds_factor=DS_FACTOR)
        t,temperature = _crop_traces(t,temperature)

    # Save the downsampled data to a mat file
    trigger_label = _get_trigger_from_SR(SR)

    _log.info('Saving outputs')
    _log.debug(f'{t.shape[0]=}\n{pdiff.shape[0]=}\n{dia_sub.shape[0]=}\n{hr_bpm.shape[0]=}\n{temperature.shape[0]=}\n{flow.shape[0]=}')

    physiol_df = pd.DataFrame()
    if has_pdiff:
        physiol_df['pdiff'] = pdiff
    if has_dia:
        physiol_df['dia'] = dia_sub 
    if has_ekg:
        physiol_df['hr_bpm'] = hr_bpm
    if has_temperature:
        physiol_df['temperature'] = temperature
    if has_flowmeter:
        physiol_df['flowmeter'] = flow

    fn_physiol = spec.to_alf('physiology','table','pqt','cibrrig',extra=trigger_label)
    fn_physiol_timestamps =  spec.to_alf('physiology','times','npy','cibrrig',extra=trigger_label)
    fn_heartbeat = spec.to_alf('heartbeat','times','npy','cibrrig',extra=trigger_label)
    physiol_df.to_parquet(save_path.joinpath(fn_physiol))
    np.save(save_path.joinpath(fn_physiol_timestamps),t)
    
    if has_ekg:
        np.save(save_path.joinpath(fn_heartbeat),heartbeats)

    fn_breath_onsets = spec.to_alf('breaths','times','npy','cibrrig',extra=trigger_label)
    fn_breaths = spec.to_alf('breaths','table','pqt','cibrrig',extra=trigger_label)

    if has_dia:
        # But strip the data referenced to the 10K sampling
        dia_df.drop(['on_samp','off_samp','duration_samp','pk_samp'],axis=1,inplace=True)

        # Compute sighs from diaphragm
        dia_df = label_sighs(dia_df)

        # Save breaths features 
        dia_df.to_parquet(save_path.joinpath(fn_breaths),index=False)

        breath_onsets = dia_df['on_sec'].values
        np.save(save_path.joinpath(fn_breath_onsets),breath_onsets)

        fn_dia_filt = spec.to_alf('diaphragm','filtered','npy','cibrrig',extra=trigger_label)
        np.save(save_path.joinpath(fn_dia_filt),dia_filt)

        fn_dia_filt_times = spec.to_alf('diaphragm','times','npy','cibrrig',extra=trigger_label)
        np.save(save_path.joinpath(fn_dia_filt_times),t_full)
    else:
        dia_df = pd.DataFrame() # Write an empty table so breathmetrics has a filename
        dia_df.to_parquet(save_path.joinpath(fn_breaths))
        np.save(save_path.joinpath(fn_breath_onsets),np.array([]))

#TODO:
# Running this not from the cibrrig preprocess folder is problematic because it needs to run the breathmetrics_proc.m file 
# At this point I don't know how best to incorporate this matlab script into a python workflow without hacking paths         
def run(session_path, v_in=9, inhale_pos=False, save_path=None,debug=False):
    '''
    Set chan to -1 if no data is recorded.
    '''
    _log.setLevel(logging.DEBUG) if debug else None

    session_path = Path(session_path)
    save_path = save_path or session_path.joinpath('alf')
    save_path.mkdir(exist_ok=True)
    ephys_path = session_path.joinpath('raw_ephys_data')
    _log.debug(f'Looking for ephys in : {ephys_path}')
    ni_fn_list = list(ephys_path.glob('*nidq.bin'))
    ni_fn_list.sort()
    _log.debug(f'Found Nidaq data: {ni_fn_list}')

    for ni_fn in ni_fn_list:
        _log.info(f'Processing {ni_fn}')
        SR = spikeglx.Reader(ni_fn)
        ni_wiring = spikeglx.get_sync_map(ni_fn.parent)
        run_one(SR,ni_wiring,v_in,inhale_pos,save_path)
        trigger_label = _get_trigger_from_SR(SR)
        has_pdiff = 'pdiff' in ni_wiring.keys()
        has_flow = 'flowmeter' in ni_wiring.keys()

        if has_pdiff or has_flow:
            _log.info('='*50)
            _log.info('Sending to Breathmetrics')
            command = ['matlab','-batch',f"breathmetrics_proc('{save_path}','{trigger_label}')"]
            try:
                subprocess.run(command,check=True,cwd=BM_PATH)
                # Rewrite using pandas for a strange compatability issue
                fn = list(save_path.glob(f'*breaths*table*{trigger_label}*.pqt'))[0]
                aa = pd.read_parquet(fn)
                aa.to_parquet(fn)
                
            except Exception as e:
                _log.error(e)
                _log.error('='*50)
                _log.error('='*15+ 'BREATHMETRICS FAILED Is it installed on the Matlab path or are you not running from the preprocess folder?!!'+ '='*15)
                _log.error('='*50)
                import time
                time.sleep(5)
                continue

        else:
            _log.info('No airflow signal so not performing BM')
        

@click.command()
@click.argument('session_path')
@click.option('-v','--v_in','v_in',default=9,type=float,show_default=True)
@click.option('-i','--inhale_pos',is_flag=True,default=False,show_default=True)
@click.option('-s','--save_path','save_path',default=None,show_default=True)
@click.option('--debug',is_flag=True)
def main(session_path, v_in, inhale_pos, save_path,debug):
    run(session_path, v_in, inhale_pos, save_path,debug)


if __name__=='__main__':
    main()
