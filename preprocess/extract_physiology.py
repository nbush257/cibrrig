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
import nidq_utils
import spikeglx
from one.alf import spec
import physiology
import logging
logging.basicConfig()
_log = logging.getLogger('downsample_and_process')
_log.setLevel(logging.INFO)


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

def run_one(SR, pdiff_chan, flowmeter_chan, dia_chan, ekg_chan, temp_chan, v_in, inhale_pos, save_path):
    DS_FACTOR=10
    # INIT output variables
    pdiff = np.array([])
    flow = np.array([])
    dia_filt= np.array([])
    dia_sub = np.array([])
    temperature = np.array([])
    heartbeats = np.array([])
    hr_bpm = np.array([])

    has_pdiff = pdiff_chan>=0
    has_flow = flowmeter_chan>=0
    has_dia = dia_chan>=0
    has_ekg = ekg_chan>=0
    has_temperature = temp_chan>=0

    if inhale_pos:
        inhale_dir=1
    else:
        inhale_dir=-1


    save_path = save_path or SR.file_bin.parent

    # LOAD Memory map from SGLX
    sr = SR.fs
    # Get tvec
    t = nidq_utils.get_tvec_from_SR(SR)
    t = t[::DS_FACTOR]
    sr_sub = sr/DS_FACTOR

    _log.info(f'Sampling rate is {sr}')
    _log.info(f'Downsampling to {sr_sub}')

    # Process diaphragm
    # Must do before explicit ekg processing because it attempts
    # to find the heartbeats, but it is not as good as the
    # explicit EKG channel and so we want to overwrite heartbeats with those data if they exist
    if has_dia:
        _log.info('Processing diaphragm')
        raw_dia,sr_dia = nidq_utils.load_dia_emg(SR,dia_chan)
        dia_df,dia_sub,sr_dia_sub,HR,dia_filt,heartbeats = nidq_utils.filt_int_ds_dia(raw_dia,sr_dia,ds_factor=DS_FACTOR)
        t,dia_sub = _crop_traces(t,dia_sub)

    # Process EKG
    if has_ekg:
        _log.info('Processing EKG')
        heartbeats = nidq_utils.extract_hr_channel(SR,ekg_chan)
        _,hr_bpm = physiology.compute_avg_hr(heartbeats,t_target=t)


    # Process PDIFF
    if has_pdiff:
        _log.info('Processing pressure diffrential sensor')
        pdiff,sr_pdiff = nidq_utils.load_ds_pdiff(SR, pdiff_chan,ds_factor=DS_FACTOR,inhale_dir=inhale_dir)
        t,pdiff = _crop_traces(t,pdiff)

    # Process Flowmeter
    if has_flow:
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
    physiol_df['t'] = t
    physiol_df['pdiff'] = pdiff if has_pdiff else None
    physiol_df['dia'] = dia_sub if has_dia else None
    physiol_df['hr_bpm'] = hr_bpm if has_ekg else None
    physiol_df['temperature'] = temperature if has_temperature else None
    physiol_df['flowmeter'] = flow if has_flow else None
    physiol_df.set_index('t',inplace=True)

    fn_physiol = spec.to_alf('physiology','traces','pqt','cibrrig',extra=trigger_label)
    fn_heartbeat = spec.to_alf('heartbeat','times','npy','cibrrig',extra=trigger_label)

    physiol_df.to_parquet(save_path.joinpath(fn_physiol))
    np.save(save_path.joinpath(fn_heartbeat),heartbeats)

    fn_breath_onsets = spec.to_alf('breaths','times','npy','cibrrig',extra=trigger_label)
    fn_breaths = spec.to_alf('breaths','features','tsv','cibrrig',extra=trigger_label)

    if has_dia:
        # Save the extracted diaphragm to a csv
        # But strip the data referenced to the 10K sampling
        dia_df.drop(['on_samp','off_samp','duration_samp','pk_samp'],axis=1,inplace=True)

        # Compute sighs from diaphragm
        dia_df = label_sighs(dia_df)

        # Save breaths features 
        dia_df.to_csv(save_path.joinpath(fn_breaths),sep='\t')

        breath_onsets = dia_df['on_sec'].values
        np.save(save_path.joinpath(fn_breath_onsets),breath_onsets)

        fn_dia_filt = spec.to_alf('diphragm','filtered','npy','cibrrig',extra=trigger_label)
        np.save(save_path.joinpath(fn_dia_filt),dia_filt)
    else:
        dia_df = pd.DataFrame() # Write an empty table so breathmetrics has a filename
        dia_df.to_csv(save_path.joinpath(fn_breaths),sep='\t')

        



@click.command()
@click.argument('session_path')
@click.option('-p','--pdiff_chan','pdiff_chan',default=4,show_default=True)
@click.option('-f','--flowmeter_chan','flowmeter_chan',default=-1,show_default=True)
@click.option('-d','--dia_chan','dia_chan',default=0,show_default=True)
@click.option('-e','--ekg_chan','ekg_chan',default=1,show_default=True)
@click.option('-t','--temp_chan','temp_chan',default=7,show_default=True)
@click.option('-v','--v_in','v_in',default=9,type=float,show_default=True)
@click.option('-i','--inhale_pos',is_flag=True,default=False,show_default=True)
@click.option('-s','--save_path','save_path',default=None,show_default=True)
@click.option('--debug',is_flag=True)
def main(session_path, pdiff_chan, flowmeter_chan, dia_chan, ekg_chan, temp_chan, v_in, inhale_pos, save_path,debug):
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
    _log.debug(f'Found Nidaq data: {ni_fn_list}')

    for ni_fn in ni_fn_list:
        has_breathmetrics=False
        _log.info(f'Processing {ni_fn}')
        SR = spikeglx.Reader(ni_fn)
        run_one(SR,pdiff_chan,flowmeter_chan,dia_chan,ekg_chan,temp_chan,v_in,inhale_pos,save_path)
        trigger_label = _get_trigger_from_SR(SR)
        if pdiff_chan>=0 or flowmeter_chan>=0:
            _log.info('='*50)
            _log.info('Sending to Breathmetrics')
            command = ['matlab','-batch',f"breathmetrics_proc('{save_path}','{trigger_label}')"]
            subprocess.run(command,check=True)
            has_breathmetrics = True

        else:
            _log.info('No airflow signal so not performing BM')
        
        # Append breathmetrics
        fn_breath_onsets = save_path.joinpath(spec.to_alf('breaths','times','npy','cibrrig',extra=trigger_label))

        if not fn_breath_onsets.exists(): # If no diaphragm (i.e. awake)
            fn_features = list(save_path.glob(f'*features.{trigger_label}*.tsv'))[0]
            df = pd.read_csv(fn_features,sep='\t')
            on_secs = df['inhale_onsets'].values
            np.save(fn_breath_onsets,on_secs)






if __name__=='__main__':
    main()
