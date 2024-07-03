'''
Wrapper to Chronux coherency computations in Matlab
'''
#TODO: Confirm the mapping of Chronux PHI to inspiration/expiration (Inspiration [0,pi), expiration [-pi,0))
import click
from pathlib import Path
import numpy as np
import re
import scipy.io.matlab as sio
import sys
import spikeinterface.full as si
import subprocess
import pandas as pd
import logging
import one.alf.io as alfio
logging.basicConfig()
_log = logging.getLogger('extract_chronux')
_log.setLevel(logging.INFO)
sys.path.append('../..') # There are some funky, circular dependencies that will need to be cleaned up


#Chronux parameters 
ERR = [[2.0,0.001]]
TAPERS = [[3.0,5.0]]
WIN=25.0
def adjust_chronux_phi(phi):
    '''
    Modify phi so that [-pi,0) is expiration and [0,pi) is inspiration.

    '''
    phi_adj = phi-(np.pi/2)
    phi_adj[phi_adj<-np.pi] +=2*np.pi
    phi_adj = -phi_adj

    return(phi_adj)


def run_chronux(spike_times,spike_clusters,cluster_ids,x,xt,t0,tf,verbose=True):
    """Run chronux on a subset of data. 
    Runs agnostic of underlying file structure.
    Submits a subprocess command to matlab, so matlab must be installed and chronux must be
    in the matlab path.

    Args:
        spike_times (1D numpy array): times in seconds of spikes
        spike_clusters (1D numpy array): clusters each spike is associated with
        cluster_ids (1D numpy array): list of unique clusters to analyse
        x (1D numpy array): continuous valued variable to compute coherence against
        xt (1D numpy array): time of each sample in x
        t0 (float): first time of window to analyse (seconds)
        tf (float): last time of window to analyse (seconds)
    """    

    # Logic of time is important here becasue chronux assumes that x starts at t = 0
    idx = np.logical_and(spike_times>t0,spike_times<tf)
    spike_times = spike_times[idx]
    spike_clusters = spike_clusters[idx]
    s0,sf = np.searchsorted(xt,[t0,tf])
    x = x[s0:sf]
    sr = 1./np.mean(np.diff(xt))

    params = {}
    params['Fs'] = sr
    params['err'] = ERR
    params['tapers'] = TAPERS
    params['win'] = WIN
    params['verbose'] = verbose

    data_out = {}
    data_out['x'] = x
    data_out['spike_times'] = spike_times - t0 # Must subtract t0 here in order to align the first sample
    data_out['spike_clusters'] = spike_clusters
    data_out['cluster_ids'] = cluster_ids
    data_out['params'] = params
    sio.savemat('.temp_chronux.mat', data_out,oned_as='column')

    command = ["matlab","-batch", "python_CHRONUX('.temp_chronux.mat');"]
    subprocess.run(command, check=True)
    chronux_rez = sio.loadmat('.temp_chronux.mat')
    _log.debug('Removing temp mat file')
    Path('.temp_chronux.mat').unlink()
    chronux_rez['params'] = params

    return(chronux_rez)


def reshape_chronux_output(chronux_rez,cluster_ids,n_total_clusters,mode = 'unit_level',min_rr=0.2):
    """Given the chronux result loaded from the mat file, reshape either into a 
    more user friendly mat file or a unit level data frame

    Mode: unit_level returns a dataframe of the coherence, bounds, and phase lags for each cluster.

    Args:
        chronux_rez (dict): result from the chronux output
        cluster_ids (1D numpy array): cluster IDS
        n_total_clusters (int): number of total clusters in the recording. Required if you did  not compute coherence on the non-QC units
        mode (str, optional): what format to output to. Defaults to 'unit level'. ('mat' or 'unit_level')
    """    
    freqs = chronux_rez['freqs'].ravel()
    # Chop the frequencies to less than 20Hz
    chop_freqs_idx = np.where(freqs<20)[0][-1]

    # Get the peak in the breathing rate. Assumes breathing rate is greater than "min_rr"
    breathing_spectrum = chronux_rez['breathing_spectrum'].ravel()
    breathing_spectrum[freqs<min_rr] = 0
    rr_sample = np.argmax(breathing_spectrum)
    rr = freqs[rr_sample]
    _log.info(f'Computed breathing rate was {rr:0.1f}Hz')

    # 
    if mode=='mat':
        save_data = {}
        save_data['full_coherence'] = chronux_rez['full_coherence'][:,:chop_freqs_idx]
        save_data['full_coherence_lb'] = chronux_rez['full_coherence_lb'][:,:chop_freqs_idx]
        save_data['full_coherence_ub'] = chronux_rez['full_coherence_ub'][:,:chop_freqs_idx]
        save_data['full_phi'] = chronux_rez['full_phi'][:,:chop_freqs_idx]
        save_data['phi_std'] = chronux_rez['full_phistd'][:,:chop_freqs_idx]
        save_data['freqs'] = freqs[:chop_freqs_idx]
        save_data['params'] = chronux_rez['params']
        save_data['cluster_id'] = cluster_ids
        return(save_data)
    elif mode=='unit_level':
        unit_level_coh = pd.DataFrame(index=np.arange(n_total_clusters))
        unit_level_coh.loc[cluster_ids,'coherence'] = chronux_rez['full_coherence'][:,rr_sample]
        unit_level_coh.loc[cluster_ids,'coherence_lb'] = chronux_rez['full_coherence_lb'][:,rr_sample]
        unit_level_coh.loc[cluster_ids,'coherence_ub'] = chronux_rez['full_coherence_ub'][:,rr_sample]
        unit_level_coh.loc[cluster_ids,'phi_std'] = chronux_rez['full_phistd'][:,rr_sample]
        unit_level_coh.loc[cluster_ids,'phi'] = chronux_rez['full_phi'][:,rr_sample]
        unit_level_coh.loc[cluster_ids,'cluster_id'] = cluster_ids
        unit_level_coh.loc[cluster_ids,'phi_adj'] = adjust_chronux_phi(unit_level_coh['phi'])
        return(unit_level_coh)
    else:
        raise ValueError(f'Mode {mode} not supported. Must be "mat" or "unit_level"')


def run_phy_probe(phy_path,t0,tf,x,xt,use_good=True,verbose=True):
    _log.info(f'Running {phy_path.name}')
    spike_times = np.load(phy_path.joinpath('spike_times.npy'))
    spike_clusters = np.load(phy_path.joinpath('spike_clusters.npy'))
    groups = pd.read_csv(phy_path.joinpath('cluster_group.tsv'),sep='\t')

    n_total_clusters = np.unique(spike_clusters).shape[0]
    if use_good:
        cluster_ids = groups['cluster_id'][groups['group']=='good'].values
        idx = np.isin(spike_clusters,cluster_ids)
        spike_times = spike_times[idx]
        spike_clusters = spike_clusters[idx]
    else:
        cluster_ids = np.unique(spike_clusters)
    

    # Run Chronux
    chronux_rez = run_chronux(spike_times,spike_clusters,cluster_ids,x,xt,t0,tf,verbose=verbose)

    # Shape results 
    unit_coh = reshape_chronux_output(chronux_rez,cluster_ids,n_total_clusters,mode='unit_level')
    
    # Save results
    fn_chronux = alfio.files.spec.to_alf('clusters','coherence','pqt')
    unit_coh.to_parquet(phy_path.joinpath(fn_chronux))




def run_probe(probe_path,t0,tf,x,xt,use_good=True,verbose=True):
    """
    Compute coherence using chronux ALF organized spike data in a probe path

    Args:
        probe_path (Pathlib path): path to the ALF spiking data
        t0 (float): start of the epoch to comute on
        tf (float): end of the epoch to compute on
        x (1D numpy array): continuous variable to compute coherence against
        xt (1D numpy array): timestamps of the x variable
        use_good (bool, optional): Flag to only compute on neurons that have been designated good. Defaults to True.
    """
    _log.info(f'Running {probe_path.name}')
    spikes = alfio.load_object(probe_path,'spikes')
    clusters = alfio.load_object(probe_path,'clusters')
    if 'coherence' in clusters.keys():
        _log.warning('Coherence already computed. Skipping')
        return
    n_total_clusters = clusters.uuids.shape[0]
    spike_times = spikes.times
    spike_clusters = spikes.clusters

    if use_good:
        cluster_ids = clusters.metrics['cluster_id'][clusters.metrics.group=='good'].values
        idx = np.isin(spikes.clusters,cluster_ids)
        spike_times = spike_times[idx]
        spike_clusters = spike_clusters[idx]
    else:
        cluster_ids = np.unique(spikes.clusters)

    # Run Chronux
    chronux_rez = run_chronux(spike_times,spike_clusters,cluster_ids,x,xt,t0,tf,verbose=verbose)

    # Shape results 
    unit_coh = reshape_chronux_output(chronux_rez,cluster_ids,n_total_clusters,mode='unit_level')
    
    # Save results
    fn_chronux = alfio.files.spec.to_alf('clusters','coherence','pqt')
    unit_coh.to_parquet(probe_path.joinpath(fn_chronux))


def run_session(session_path,t0,tf,var='dia',use_good=True,verbose=True):
    """
    Run chronux coherence extraction on all probes in a session
    Should have the "physiology" object extracted, and all probes should be in ALF format.

    Args:
        session_path (Pathlib Path): Path to the session.
        t0 (float): start of the epoch to comute on
        tf (float): end of the epoch to compute on
        x (1D numpy array): continuous variable to compute coherence against
        xt (1D numpy array): timestamps of the x variable
        use_good (bool, optional): Flag to only compute on neurons that have been designated good. Defaults to True.
    """    
    _log.info(f'\nComputing coherence for {session_path}.\n\t{t0=}\n\t{tf=}\n\t{var=}\n\t{use_good=}')
    physiology_data = alfio.load_object(session_path.joinpath('alf'),'physiology',short_keys=True)
    x = physiology_data[var]
    xt = physiology_data['times']
    probe_paths = list(session_path.joinpath('alf').glob('probe[0-9][0-9]'))
    for probe in probe_paths:
        run_probe(probe,t0,tf,x,xt,use_good=use_good,verbose=verbose)


@click.command()
@click.argument('session_path')
@click.option('--t0',default=0,show_default=True,help = 'Time in seconds of the beginning of the window to consider in the coherence computation.')
@click.option('--tf',default=300,show_default=True,help = 'Time in seconds of the end of the window to consider in the coherence computation.')
@click.option('--var',default='dia',show_default=True,help ='Which variable to compute coherence against')
@click.option('--include_all','-i',is_flag=True,help='If set, runs on all units. Default behavior is to run only on the units identified as "good"')
@click.option('--verbose','-v',is_flag=True,help='Updates user of computation for each unit')
def main(session_path,t0,tf,var,include_all,verbose):
    session_path = Path(session_path)
    use_good = not include_all
    run_session(session_path,t0,tf,var,use_good,verbose)

if __name__=='__main__':
    main()