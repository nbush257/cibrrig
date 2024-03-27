'''
Should apply sync and export to alf for all probes given a session path
'''

from ibllib.ephys.spikes import ks2_to_alf,apply_sync
from pathlib import Path
import spikeglx
import one.alf.io as alfio
import pandas as pd
import logging
import numpy as np
import click
logging.basicConfig()
_log = logging.getLogger('convert_to_alf')
_log.setLevel(logging.INFO)

def get_metrics_from_si(ks4_dir):
    """
    Aggregate all the cluster level metrics from the kilosort directory

    Args:
        ks4_dir (Pathlib Path): Kilosort directory
    """    
    metrics_files = ks4_dir.glob('*.tsv')
    metrics = pd.DataFrame()
    for fn in metrics_files:
        df = pd.read_csv(fn,sep='\t')
        if 'cluster_id' not in df.columns:
            continue
        if 'cluster_id' in metrics.columns:
            metrics = metrics.merge(df,on='cluster_id',how='outer')
        else:
            metrics = df
    return(metrics)


def save_metrics(metrics,out_path):
    """
    Convinience function to save the metrics
    Args:
        metrics (_type_): _description_
        out_path (_type_): _description_
    """    
    metrics_fn = alfio.spec.to_alf('clusters','metrics','pqt')
    metrics.to_parquet(out_path.joinpath(metrics_fn))


def get_ap_breaks_samps(ap_files):
    """
    Return a numpy array of the cumulative sample length for each recording if there are multiple triggers
    Example: recording 1 has 110 samples and recording 2 has 24 samples. Returns: [0,110,134]

    Args:
        ap_files (_type_): _description_
    """    
    breaks = [0]
    for fn in ap_files:
        SR = spikeglx.Reader(fn)
        breaks.append(breaks[-1]+SR.ns)
    breaks = np.array(breaks)
    return(breaks)



def sync_spikes(ap_files,spikes):
    """Finds the computed synchronization files that
    line up the IMEC clock to the NIDAQ clock and adjusts the spikes to the 
    NIDAQ clock. Spikes that occur at negative timea re set to zero

    Args:
        ap_files (_type_): list of all the ap binary files
        spikes (_type_): spikes alf object
    """    
    breaks_samps = get_ap_breaks_samps(ap_files)
    rec_idx = np.searchsorted(breaks_samps,spikes.samples)-1
    all_times_adj = []
    for ii,ap_file in enumerate(ap_files):     
        parts = alfio.files.filename_parts(ap_file.name)
        sync_file = alfio.filter_by(ap_file.parent,object=parts[1],extra='sync')[0][0]
        sync_file = ap_file.parent.joinpath(sync_file)
        this_rec_times= spikes.times[rec_idx==ii]
        times_adj = apply_sync(sync_file,this_rec_times)
        all_times_adj.append(times_adj)
    all_times_adj = np.concatenate(all_times_adj)

    all_times_adj[all_times_adj<0] = 0

    return(all_times_adj)

    
@click.command()
@click.argument('session_path')
def main(session_path):
    session_path = Path(session_path)
    ephys_path = session_path.joinpath('raw_ephys_data')
    probes_alfs = list(session_path.joinpath('alf').glob('probe[0-9][0-9]'))

    for probe_alf in probes_alfs:
        _log.info(f'Converting {probe_alf.name}')

        # Get paths
        ks4_dir = probe_alf.joinpath('kilosort4')
        bin_path = ks4_dir.joinpath('recording.dat')
        out_path = probe_alf

        # Convert to ALF
        ks2_to_alf(ks4_dir,bin_path,out_path)

        # Get metrics
        _log.info('Extracting spike metrics')
        metrics = get_metrics_from_si(ks4_dir)
        save_metrics(metrics,out_path)

        # Get ap files for synchronizing
        _log.info('Syncronizing to NIDAQ clock')
        raw_probe = list(ephys_path.glob(probe_alf.name))
        assert(len(raw_probe)==1),f'More than one path in {ephys_path} matches {probe_alf.name}'
        raw_probe = raw_probe[0]
        ap_files = list(raw_probe.rglob('*ap.bin'))
        _log.info(f'Found {len(ap_files)} ap files')

        # Adjust spikes from the IMEC clock to the NIDAQ clock
        spikes = alfio.load_object(out_path,'spikes')
        times_adj = sync_spikes(ap_files,spikes)
        
        # Output
        _log.info("Saving adjusted and old spike times")
        times_old_fn = alfio.files.spec.to_alf(object='spikes',attribute='times',timescale='ephysClock',extension='npy')
        times_adj_fn= alfio.files.spec.to_alf(object='spikes',attribute='times',extension='npy')
        np.save(out_path.joinpath(times_old_fn),spikes.times)
        np.save(out_path.joinpath(times_adj_fn),times_adj)
        

if __name__ == '__main__':
    main()