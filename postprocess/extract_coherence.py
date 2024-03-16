'''
Wrapper to Chronux coherency computations in Matlab
'''
#TODO: Confirm the mapping of Chronux PHI to inspiration/expiration (Inspiration [0,pi), expiration [-pi,0))
import click
import spikeglx
from pathlib import Path
import numpy as np
import re
import scipy.io.matlab as sio
import sys
import spikeinterface.full as si
import subprocess
import pandas as pd
import logging
from one.alf import spec
logging.basicConfig()
_log = logging.getLogger('_extract_coherence_')
_log.setLevel(logging.INFO)
sys.path.append('../..') # There are some funky, circular dependencies that will need to be cleaned up
import iso_npx.utils


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


def run_chronux(spike_times,spike_clusters,cluster_ids,x,xt,t0,tf):
    """Run chronux on a subset of data. 
    Runs agnostic of underlying file structure.
    Submits a subprocess command to matlab, so matlab must be installed and chronux must be
    in the matlab path.

    Args:
        spike_times (_type_): times in seconds of spikes
        spike_clusters (_type_): clusters each spike is associated with
        cluster_ids (_type_): list of unique clusters to analyse
        x (_type_): continuous valued variable to compute coherence against
        xt (_type_): time of each sample in x
        t0 (_type_): first time of window to analyse (seconds)
        tf (_type_): last time of window to analyse (seconds)
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


def reshape_chronux_output(chronux_rez,cluster_ids,mode = 'mat'):
    """Given the chronux result loaded from the mat file, reshape either into a 
    more user friendly mat file or a unit level data frame

    Args:
        chronux_rez (_type_): _description_
        cluster_ids (_type_): _description_
        mode (str, optional): _description_. Defaults to 'mat'. ('mat' or 'unit_level')
    """    
    freqs = chronux_rez['all_f'].ravel()
    chop_freqs_idx = np.where(freqs<20)[0][-1]

    if mode=='mat':
        save_data = {}
        save_data['full_coherence'] = chronux_rez['full_coherence'][:,:chop_freqs_idx]
        save_data['full_coherence_lb'] = chronux_rez['full_coherence_lb'][:,:chop_freqs_idx]
        save_data['full_coherence_ub'] = chronux_rez['full_coherence_ub'][:,:chop_freqs_idx]
        save_data['full_phi'] = chronux_rez['full_phi'][:,:chop_freqs_idx]
        save_data['freqs'] = freqs[:chop_freqs_idx]
        save_data['params'] = chronux_rez['params']
        return(save_data)
    elif mode=='unit_level':
        coh = chronux_rez['full_coherence']
        max_coh_idx = np.argmax(coh,axis=1)
        xx = np.arange(len(cluster_ids))
        unit_level_coh = {}
        unit_level_coh['coherence'] = chronux_rez['full_coherence'][xx,max_coh_idx]
        unit_level_coh['coherence_lb'] = chronux_rez['full_coherence_lb'][xx,max_coh_idx]
        unit_level_coh['coherence_ub'] = chronux_rez['full_coherence_ub'][xx,max_coh_idx]
        unit_level_coh['phi'] = chronux_rez['full_phi'][xx,max_coh_idx]
        unit_level_coh['cluster_id'] = cluster_ids
        unit_level_coh['phi_adj'] = adjust_chronux_phi(unit_level_coh['phi'])
        df = pd.DataFrame(unit_level_coh)
        return(df)
    else:
        raise ValueError(f'Mode {mode} not supported. Must be "mat" or "unit_level"')


def run_on_phy(gate_path,phy_path,t0,tf,var='dia',use_good=True):
    """Runs chronux given a gate path (which contains auxiliary data) and a phy path.

    Args:
        gate_path (_type_): _description_
        phy_path (_type_): _description_
        t0 (_type_): _description_
        tf (_type_): _description_
        var(str): Variable to compute coherence against. Defaults to "dia" 
        use_good (bool, optional): _description_. Defaults to True. - Whether to only compute for units catagorized as "good" in sorting
    """    

    save_fn_mat = phy_path.joinpath(spec.to_alf('clusters','coherence','mat','_chronux_'))
    save_fn_unitlevel = phy_path.joinpath(spec.to_alf('clusters','coherence','tsv','_chronux_'))
    # Load diaphragm
    _log.info('Loading aux')
    epochs,breaths,aux = iso_npx.utils.load_aux(gate_path) # This needs to get moved to a more reasonable location.
    x = aux[var].ravel()
    aux_t = aux['t'].ravel()


    _log.info('Loading spiking')
    phy_model = si.PhySortingExtractor(phy_path)
    sv = phy_model.to_spike_vector()
    spike_times = sv['sample_index']/phy_model.sampling_frequency
    spike_clusters = sv['unit_index']


    group =  phy_model.get_property('quality')

    # THere may become a bug here with there being no spikes associated with a cluster ID 
    # That may arise from the subsetting
    if use_good:
        _log.debug('Only using good units')
        cluster_ids = np.where(group=="good")[0]
    else:
        cluster_ids = np.unique(spike_clusters)

    # RUN CHRONUX
    _log.info('Sending to chronux')
    chronux_rez = run_chronux(spike_times,spike_clusters,cluster_ids,x,aux_t,t0,tf)

    # Shape results 
    save_mat = reshape_chronux_output(chronux_rez,cluster_ids,mode='mat')
    unit_coh = reshape_chronux_output(chronux_rez,cluster_ids,mode='unit_level')

    # Save results
    sio.savemat(save_fn_mat,save_mat)
    unit_coh.to_csv(save_fn_unitlevel,sep='\t',index=False)

@click.command()
@click.argument('gate_path')
@click.option('--t0',default=0,show_default=True,help = 'Time in seconds of the beginning of the window to consider in the coherence computation.')
@click.option('--tf',default=300,show_default=True,help = 'Time in seconds of the end of the window to consider in the coherence computation.')
@click.option('--var',default='dia',show_default=True,help ='Which variable to compute coherence against')
@click.option('--include_all','-i',is_flag=True,help='If set, runs on all units. Default behavior is to run only on the units identified as "good"')
def main(gate_path,t0,tf,var,include_all):
    gate_path = Path(gate_path)
    # Get all phy folders: only coded now for spikeinterface structure with KS3
    phy_list_spikeinterface = list(gate_path.glob('si-sort/*/phy_output')) 
    if len(phy_list_spikeinterface) ==0:
        _log.warning('No sorted data found. Exiting')
    #TODO: add potentially other folder structures
    use_good = not include_all
    for phy_path in phy_list_spikeinterface:
        _log.info(f'Running on\n\t{phy_path=}\n\t{gate_path=}')
        run_on_phy(gate_path,phy_path,t0,tf,var,use_good = use_good)

if __name__=='__main__':
    main()