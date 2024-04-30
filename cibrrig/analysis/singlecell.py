from ..preprocess.physiology import compute_dia_phase
import numpy as np
import warnings
from tqdm import tqdm


def get_phase_curve(ts,breaths,phi_t=None,phi=None,nbins=100):
    '''
    Computes the firing rate of a neuron for a given respiratory  phase

    Computes on a per-breath basis in order to get a distribution
    

    :param ts: Times of spikes for a single neuron
    :param breaths: Breaths bunch
    :param phi_t: time basis of the phase variable phi
    :param phi: Repiratory phase at a given time. 
    :param nbins: number of bins in the phasic PSTH (defaults to 100 bins)
    :return: out_dict with keys-- bins,rate_mean,rate_std,rate_sem
    '''

    # Unpack the breath onsets
    ons = breaths.on_sec
    offs = breaths.off_sec
    nbreaths = len(ons)

    # Compute phase
    if phi_t is None or phi is None:
        phi_t, phi = compute_dia_phase(ons, offs)
    dt = np.mean(np.diff(phi_t))

    # Compute the bins
    bins = np.linspace(-np.pi,np.pi,nbins+1)

    # Get the samples of phi that are the onsets of breaths (should be zerocrossings by definition)
    on_samps = np.searchsorted(phi_t,ons)

    # map spiketimes into samples of phi. Create a binary spike train like Phi to use as a mask
    spikesamps = np.searchsorted(phi_t,ts)-1 # need to subtract one to not overrun the slice
    btrain = np.zeros_like(phi).astype('bool')
    btrain[spikesamps] = 1

    # Preallocate the rate matix (Nbins x Nbreaths)
    # each entry will be the number of spikes per bin for a given breath, so not actually a rate
    rate = np.zeros([nbins,nbreaths-1])
    prior = np.zeros([nbins,nbreaths-1])
    posterior = np.zeros([nbins,nbreaths-1])

    # Loop over every breath and make a histogram
    for ii in range(len(ons)-1):
        on = on_samps[ii]
        on_next = on_samps[ii+1]
        phi_slice = phi[on:on_next]
        btrain_slice = btrain[on:on_next]

        # Compute the prior (phase) and posterior (phase given spike) histograms
        prior[:,ii] = np.histogram(phi_slice,bins)[0]*dt # scale by time sampling
        posterior[:,ii] = np.histogram(phi_slice[btrain_slice],bins)[0]


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rate = np.divide(posterior,prior)

    # center the bin edges
    bins = bins[:-1] + np.diff(bins)[0]

    # Compute means/std/sem
    rate_mean = np.nanmean(rate,1)
    rate_std = np.nanstd(rate,1)
    rate_sem = np.nanstd(rate,1)/np.sqrt(nbreaths)

    # Package
    out_dict = {}
    out_dict['bins'] = bins
    out_dict['rate_mean'] = rate_mean
    out_dict['rate_std'] = rate_std
    out_dict['rate_sem'] = rate_sem
    out_dict['rate_lb'] = rate_mean - rate_sem
    out_dict['rate_ub'] = rate_mean + rate_sem

    return(out_dict)


def get_all_phase_curves(spike_times,spike_clusters,cluster_ids,breaths,nbins=100):
    '''
    Computes the firing rate of all neurons in "cluster_id for a given respiratory  phase

    Computes on a per-breath basis in order to get a distribution
    

    :param spike_times: Times of spikes for all neurons
    :param spike_clusters: Cluster assignment of spikes for all neurons
    :param cluster_ids: id of clusters to analyse
    :param breaths: Breaths bunch
    :param nbins: number of bins in the phasic PSTH (defaults to 100 bins)

    Returns;
    bins,rate,sem. rate and sem are [nbins x n_cluster_ids]
    '''
    phi_t, phi = compute_dia_phase(breaths.on_sec, breaths.off_sec)
    
    all_rate = np.zeros([nbins,cluster_ids.shape[0]])
    all_sem = np.zeros([nbins,cluster_ids.shape[0]])
    for ii,clu in enumerate(tqdm(cluster_ids)):
        ts = spike_times[spike_clusters == clu]
        rez = get_phase_curve(ts,breaths,phi_t,phi,nbins=nbins)
        all_rate[:,ii] = rez['rate_mean']
        all_sem[:,ii] = rez['rate_sem']
    return(rez['bins'],all_rate,all_sem)


