'''
Functions to work with optogenetic stimulations, and in particular, tags. 
When run from command line will perform salt tag statistics
'''
import numpy as np
import scipy.io.matlab as sio
from pathlib import Path
import pandas as pd
from brainbox import singlecell
import subprocess
import click
import matplotlib.pyplot as plt
import seaborn as sns
import one.alf.io as alfio
from tqdm import tqdm
import sys
if sys.platform=='linux':
    import matplotlib
    matplotlib.use('TkAgg')

WAVELENGTH_COLOR = {635:'#ff3900',473:'#00b7ff'}
SALT_P_CUTOFF = 0.001
MIN_PCT_TAGS_WITH_SPIKES = 33
RATIO = 2

def compute_pre_post_raster(spike_times,spike_clusters,cluster_ids,stim_times,stim_duration = None, window_time = 0.5,bin_size=0.001,mask_dur = 0.002): 
    """Creates the rasters pre and post stimulation time. 
    Optionally blanks periods around onset and offset of light to zero (default behavior)

    Args:
        spike_times (_type_): _description_
        spike_clusters (_type_): _description_
        cluster_ids (_type_): _description_
        stim_times (_type_): _description_
        stim_duration (_type_, optional): _description_. Defaults to None.
        window_time (float, optional): _description_. Defaults to 0.5.
        bin_size (float, optional): _description_. Defaults to 0.001.
        mask_dur (float, optional): _description_. Defaults to 0.002.
    """       
    pre_raster,pre_tscale = singlecell.bin_spikes2D(spike_times,
                                            spike_clusters,
                                            cluster_ids,
                                            align_times=stim_times,
                                            pre_time=window_time+mask_dur,
                                            post_time=-mask_dur, # Calculate until 2 ms before the stimulus
                                            bin_size=bin_size,
                                            )
                                            
    post_raster,post_tscale = singlecell.bin_spikes2D(spike_times,
                                        spike_clusters,
                                        cluster_ids,
                                        align_times=stim_times,
                                        pre_time=-mask_dur, # ignore 2ms after stimulus onset to avoid artifactual spikes
                                        post_time=window_time+mask_dur,
                                        bin_size=bin_size,
                                        )
    
    # if stim_duration exists, remove any spikes within - 1 ms and + mask_duration of offset time
    if stim_duration is not None:
        stim_offsets_samp = np.searchsorted(post_tscale,stim_duration)
        post_raster[:,:,stim_offsets_samp-1:stim_offsets_samp+int(mask_dur/bin_size)] = 0

    return(pre_raster,post_raster)


def run_salt(spike_times,spike_clusters,cluster_ids,stim_times,window_time = 0.5,stim_duration= None,consideration_window=0.01):
    """    
    Runs the Stimulus Associated Latency Test (SALT - See Kvitsiani 2013) 
    on given units.

    Must pass data to matlab, and does so via saving to a temporary mat file.
    Automatically deletes the temporary mat file.

    Args:
        spike_times (np.array): times in  seconds of every spike 
        spike_clusters (np.array): cluster assignments of every spike
        cluster_ids (np.array): cluster ids to use.
        stim_times (np.array): onset times of the optogenetic stimulus
        window_time (float): duration in seconds pre and post to make the rasters. [Default = 0.5]
        stim_duration (float): duration of stimulus, used to mask the offset artifact
        consideration_window (float): Post-stimulus window to consider in the SALT tagging.  10ms is good for ChR2, longer is porbably needed for the slower ChRmine [Default = 0.01]

    Returns:
        p_stat: Resulting P value for the Stimulus-Associated spike Latency Test.
        I_stat: Test statistic, difference between within baseline and test-to-baseline information distance values. _description_
    
    """    

    pre_raster,post_raster = compute_pre_post_raster(spike_times,
                                                     spike_clusters,
                                                     cluster_ids,
                                                     stim_times,
                                                     window_time = window_time,
                                                     stim_duration = stim_duration,
                                                     bin_size = 0.001,
                                                     mask_dur=0.002)  

    #Sanitize raster
    if np.any(pre_raster<0) | np.any(post_raster<0):
        print('Warning: spike counts less than zero found. Setting to 0')
        pre_raster[pre_raster<0] = 0 
        post_raster[post_raster<0] = 0

    if np.any(pre_raster>1) | np.any(post_raster>1):
        print(f'Warning: Multiple spikes in a single ms bin found (max is {max(np.max(pre_raster),np.max(post_raster))}). Truncating to 1')
        pre_raster[pre_raster>1] = 1
        post_raster[post_raster>1] = 1

    dat_mat = {}
    dat_mat['pre_raster'] = pre_raster
    dat_mat['post_raster'] = post_raster
    dat_mat['cluster_ids'] = cluster_ids
    dat_mat['consideration_window'] = consideration_window
    sio.savemat('.temp_salt.mat',dat_mat)
    
    command = ["matlab","-batch", "python_SALT('.temp_salt.mat');"]
    subprocess.run(command, check=True)
    
    salt_rez = sio.loadmat('.temp_salt.mat')
    Path('.temp_salt.mat').unlink()
    p_stat = salt_rez['p_stat']
    I_stat = salt_rez['I_stat']

    return p_stat,I_stat


def compute_tagging_summary(spike_times,spike_clusters,cluster_ids,stim_times,window_time = 0.01,bin_size=0.001):
    """Computes the number of stims that a spike was observed and the pre and post stim spike rate

    Args:
        spike_times (1D numpy array): _description_
        spike_clusters (1D numpy array): _description_
        cluster_ids (1D numpy array): _description_
        stim_times (1D numpy array): _description_
        window_time (float, optional): _description_. Defaults to 0.01.
        bin_size (float, optional): _description_. Defaults to 0.001.
    
    Returns:
        n_responsive_stims (1D numpy array): Number of stimulations that evoked at least one spike
        pre_spikerate (1D numpy array): Spike rate in the window before stimulus onset
        post_spikerate (1D numpy array): Spike rate in the windw after stimulus onset
    """    
    n_stims = stim_times.shape[0]
    pre_spikecounts,post_spikecounts = compute_pre_post_raster(spike_times,
                                                     spike_clusters,
                                                     cluster_ids,
                                                     stim_times,
                                                     window_time =window_time,
                                                     bin_size=bin_size)
    pre_spikecounts = pre_spikecounts.sum(2)
    post_spikecounts = post_spikecounts.sum(2)

    n_responsive_stims = np.sum(post_spikecounts.astype('bool'),0)
    pre_spikerate = np.sum(pre_spikecounts,0)/n_stims/window_time
    post_spikerate = np.sum(post_spikecounts,0)/n_stims/window_time

    return(n_responsive_stims,pre_spikerate,post_spikerate)
    

def extract_tagging_from_logs(log_df,laser,verbose=True):
    """Finds the opto stims that are associated with a logged tagging episode.

    Args:
        log_df (pandas dataframe): Log from the experiment should be an autogenerated tsv
        opto_df (pandas dataframe): Opto stim dataframe extracted from the analog trace. N.B.: must be synchronized 
        verbose (bool, optional): verbose. Defaults to True.

    Raises:
        NotImplementedError: _description_
        ValueError: _description_
    
    Returns (pandas dataframe): Subdataframe from opto_df that only has the tagging data.
    """    


    # Extract the tagging start and end
    tag_epoch = log_df.query('label == "opto_tagging"')
    tag_starts = tag_epoch['start_time'].values
    tag_ends = tag_epoch['end_time'].values

    # Handle if more than one or less than 1 tagging episodes were found
    if len(tag_starts)>1:
        raise NotImplementedError("More than one tagging episode has not been implemetned yet")
    elif len(tag_starts)==0:
        raise ValueError("No Tagging episodes found.")
    else:
        pass

    # Subset (add a little buffer time so as not to miss first or last stim)
    tag_start = tag_starts[0]-0.5
    tag_end = tag_ends[0]+0.5

    #slice
    idx = np.logical_and(laser.intervals[:,0]>tag_start,laser.intervals[:,1]<tag_end)
    tags = alfio.AlfBunch()
    for k in laser.keys():
        tags[k] = laser[k][idx]
    dur_mean = np.diff(tags.intervals,1).mean()

    #Verbose and return
    print(f'Found {tags.intervals.shape[0]} tag stimulations with average duration {dur_mean:0.02}s') if verbose else None
    return(tags)
    

def make_plots(spike_times,spike_clusters,cluster_ids,tags,save_folder,salt_rez=None,pre_time=None,post_time=None,wavelength = 473,consideration_window=0.01,cmap=None):
    """ Plots rasters and PETHs for each cell aligned to 

    Args:
        spike_times (_type_): _description_
        spike_clusters (_type_): _description_
        cluster_ids (_type_): _description_
        tags (_type_): _description_
        save_folder (_type_): _description_
        salt_rez (_type_, optional): _description_. Defaults to None.
        pre_time (float, optional): _description_. Defaults to 0.05.
        post_time (float, optional): _description_. Defaults to 0.05.
        wavelength (int, optional): _description_. Defaults to 473.
        cmap (_type_, optional): _description_. Defaults to None.
    """    
    pre_time = pre_time or consideration_window*2
    post_time = post_time or consideration_window *2
    cmap = cmap or 'magma'
    if not save_folder.exists():
        save_folder.mkdir()
    else:
        print('Removing old figures.')
        for fn in save_folder.glob('*.png'):
            fn.unlink()
    stim_duration = np.diff(tags.intervals,1).mean()
    stim_times = tags.intervals[:,0]
    n_stims = stim_times.shape[0]
    if wavelength==635:
        bin_size = 0.005
    else:
        bin_size=0.0025
    peths_fine,raster_fine = singlecell.calculate_peths(spike_times,spike_clusters,cluster_ids,
                                        stim_times,
                                        pre_time=pre_time,
                                        post_time=stim_duration+post_time,
                                        bin_size=0.001,
                                        smoothing=0)

    peths,_ = singlecell.calculate_peths(spike_times,spike_clusters,cluster_ids,
                                        stim_times,
                                        pre_time=pre_time,
                                        post_time=stim_duration+post_time,
                                        bin_size=bin_size,
                                        smoothing=0)
    
    stim_no,clu_id,sps = np.where(raster_fine)
    spt = peths_fine['tscale'][sps]

    for ii,clu in enumerate(tqdm(cluster_ids,desc='Making plots')):
        plt.close('all')

        # Set up plot
        f,ax = plt.subplots(nrows=2,figsize=(4,4),sharex=True)

        # Plot data
        ax[0].vlines(spt[clu_id==ii],stim_no[clu_id==ii]-0.25,stim_no[clu_id==ii]+0.25,color='k',lw=1)
        ax[1].plot(peths['tscale'],peths['means'][ii],color='k')
        lb = peths['means'][ii]-peths['stds'][ii]/np.sqrt(n_stims)
        ub = peths['means'][ii]+peths['stds'][ii]/np.sqrt(n_stims)
        ax[1].fill_between(peths['tscale'],lb,ub,alpha=0.3,color='k')

        # Plot stim limits
        for aa in ax:
            aa.axvspan(0,stim_duration,color=WAVELENGTH_COLOR[wavelength],alpha=0.25)
            aa.axvline(0,color='c',ls=':',lw=1)
            aa.axvline(stim_duration,color='c',ls=':',lw=1)
            aa.axvline(consideration_window,color='k',ls=':',lw=1)

        # Formatting
        ax[0].set_ylim([0,n_stims])
        ax[0].set_yticks([0,n_stims])
        ax[0].set_ylabel('Stim #')
        ax[1].set_ylabel('F.R. (sp/s)')
        ax[1].set_xlabel('Time (s)')
        ax[0].set_title(f'Cluster {clu}')
        
        # Additional info if available
        if salt_rez is not None:
            this_cell = salt_rez.query('cluster_id==@clu')
            p_tagged = this_cell["salt_p_stat"].values[0]
            base_rate = this_cell["base_rate"].values[0]
            stim_rate = this_cell["stim_rate"].values[0]
            ax[0].text(0.8,0.8,f'{p_tagged=:0.03f}\n{base_rate=:0.01f} sps\n{stim_rate=:0.01f} sps',ha='left',va='center',transform=ax[0].transAxes)
        
        # Tidy axes
        sns.despine()
        plt.tight_layout()

        # Save plot
        is_tagged = False
        if salt_rez is not None:
                is_tagged = this_cell['is_tagged'].values[0]
        if is_tagged:
            save_fn = save_folder.joinpath(f'tagged_clu_{clu:04.0f}.png')
        else:
            save_fn = save_folder.joinpath(f'untagged_clu_{clu:04.0f}.png')
        plt.savefig(save_fn,dpi=300,transparent=True)
        plt.close('all')

    # Population plots - seperate by salt_p_stat <0.001
    f,ax = plt.subplots(figsize=(8,8),ncols=2,sharex=True)
    tagged_clus = np.where(np.isin(cluster_ids,salt_rez.query('is_tagged')['cluster_id'].values))[0]
    untagged_clus = np.where(np.isin(cluster_ids,salt_rez.query('~is_tagged')['cluster_id'].values))[0]
    max_spikes = 250
    cc1 = ax[0].pcolormesh(peths.tscale,np.arange(untagged_clus.shape[0]),peths.means[untagged_clus],vmin=0,vmax=max_spikes,cmap=cmap)
    cc2 = ax[1].pcolormesh(peths.tscale,np.arange(tagged_clus.shape[0]),peths.means[tagged_clus],vmin=0,vmax=max_spikes,cmap=cmap)

    for aa in ax:
        aa.axvline(0,color='w',ls=':')
        aa.axvline(consideration_window,color='silver',ls=':')
        aa.set_ylabel('Units (unordered)')
        aa.set_xlabel('Time (s)')
        if stim_duration is not None:
            aa.axvline(stim_duration,color='w',ls=':')
    ax[0].set_title('Untagged')
    ax[1].set_title(f'Tagged (salt p<{SALT_P_CUTOFF} and \nstims with  spikes>{MIN_PCT_TAGS_WITH_SPIKES:0.0f}%)')
    cax1 = plt.colorbar(cc1)
    cax2 = plt.colorbar(cc2)
    cax1.set_ticks([0,100,200,250])
    cax1.set_ticklabels(['0','100','200','>250'])
    cax2.set_ticks([0,100,200,250])
    cax2.set_ticklabels(['0','100','200','>250'])
    cax1.set_label('F.R. (sp/s)')
    cax2.set_label('F.R. (sp/s)')
    plt.tight_layout()
    plt.savefig(save_folder.joinpath('population_tags.png'),dpi=300,transparent=True)


def run_probe(probe_path,tags,consideration_window,wavelength,plot=False):
    spikes = alfio.load_object(probe_path,'spikes')
    clusters = alfio.load_object(probe_path,'clusters')
    cluster_ids = clusters.metrics['cluster_id'][clusters.metrics.group=='good'].values
    idx = np.isin(spikes.clusters,cluster_ids)
    spike_times = spikes.times[idx]
    spike_clusters = spikes.clusters[idx]



    tag_duration = np.mean(np.diff(tags.intervals,1))
    n_tags = tags.intervals.shape[0]
    tag_onsets = tags.intervals[:,0]
    # Compute SALT data
    p_stat,I_stat = run_salt(spike_times,spike_clusters,cluster_ids,tags.intervals[:,0],
                             stim_duration=tag_duration,
                             consideration_window=consideration_window)
    
    # Compute heuristic data
    n_stims_with_spikes,base_rate,stim_rate = compute_tagging_summary(spike_times,spike_clusters,cluster_ids,tag_onsets,window_time=consideration_window)

    # Export to a pqt
    salt_rez = pd.DataFrame()
    salt_rez['cluster_id']=clusters.metrics.cluster_id
    salt_rez.loc[cluster_ids,'salt_p_stat'] = p_stat
    salt_rez.loc[cluster_ids,'salt_I_stat'] = I_stat
    salt_rez.loc[cluster_ids,'n_stims_with_spikes'] = n_stims_with_spikes
    salt_rez.loc[cluster_ids,'pct_stims_with_spikes'] = n_stims_with_spikes/n_tags * 100
    salt_rez.loc[cluster_ids,'base_rate'] = base_rate
    salt_rez.loc[cluster_ids,'stim_rate'] = stim_rate
    salt_rez['is_tagged'] = False
    if wavelength == 473:
        salt_rez.loc[cluster_ids,'is_tagged'] = salt_rez.eval('salt_p_stat<@SALT_P_CUTOFF & pct_stims_with_spikes>@MIN_PCT_TAGS_WITH_SPIKES')
    elif wavelength == 635:
        salt_rez.loc[cluster_ids,'is_tagged'] = salt_rez.eval('salt_p_stat<@SALT_P_CUTOFF & pct_stims_with_spikes>@MIN_PCT_TAGS_WITH_SPIKES & stim_rate/base_rate>@RATIO')

    save_fn = probe_path.joinpath(alfio.spec.to_alf('clusters','optotag',namespace='salt',extension='pqt'))
    salt_rez.to_parquet(save_fn)
    is_tagged = {'isTagged':salt_rez['is_tagged'].values}
    alfio.save_object_npy(probe_path,is_tagged,'clusters',namespace='salt')
    print(f'optotagging info saved to {save_fn}.')

    if plot:
        make_plots(spike_times,spike_clusters,cluster_ids,tags,
                   save_folder=probe_path.joinpath('tag_plots'),
                   salt_rez=salt_rez,
                   wavelength=wavelength,
                   consideration_window=consideration_window)
    
    import json
    fn_parameters = probe_path.joinpath(alfio.spec.to_alf('optotag','parameters','json','salt'))
    params = dict(SALT_P_CUTOFF=SALT_P_CUTOFF,MIN_PCT_TAGS_WITH_SPIKES=MIN_PCT_TAGS_WITH_SPIKES,consideration_window=consideration_window,wavelength=wavelength)
    if wavelength==635:
        params['RATIO'] = RATIO
    with open(probe_path.joinpath(fn_parameters),'w') as fid:
        json.dump(params,fid)


# TODO: Implement functionality on data from multiple recordings
# TODO: Implement plotting
# TODO: Implement Chrmine option (slower optotagging responses)
@click.command()
@click.argument('session_path') # Will not work on concatnated data yet.
@click.option('-w','--consideration_window',default=0.01,help ='Option to change how much of the stimulus time to consider as important. Longer times may be needed for ChRmine')
@click.option('-l','--wavelength',default=473,help ='set wavelength of light (changes color of plots.)')
@click.option('-p','--plot',is_flag=True,help='Flag to make plots for each cell')
def main(session_path,consideration_window,plot,wavelength):
    session_path = Path(session_path)

    # Load opto times and logs
    log_fn = list(session_path.glob('*log*.tsv'))
    assert(len(log_fn)==1),f'Number of log files found was {len(log_fn)}. Should be one'
    log_fn = log_fn[0]
    #TODO: deal with multiple triggers (by concatenating previously)
    laser = alfio.load_object(session_path.joinpath('alf'),'laser',short_keys=True)
    log_df = pd.read_csv(log_fn,index_col=0,sep='\t')

    # Extract only tag times
    tags = extract_tagging_from_logs(log_df,laser)

    probe_paths = list(session_path.joinpath('alf').glob('probe[0-9][0-9]'))
    for probe in probe_paths:
        run_probe(probe,tags,consideration_window=consideration_window,wavelength=wavelength,plot=plot)
 


if __name__ == '__main__':
    main()