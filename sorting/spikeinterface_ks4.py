'''
Run Kilosort 4 locally on the NPX computer. 
Data must be reorganized using the preprocess.ephys_data_to_alf.py script or it will not work
'''
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre 
import spikeinterface.sorters as ss 
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp 
import spikeinterface.sortingcomponents.motion_interpolation as sim
import spikeinterface.full as si
from pathlib import Path
from brainbox.metrics import single_units
import numpy as np
import pandas as pd
from ibllib.ephys import spikes
from brainbox.plot import driftmap
import matplotlib.pyplot as plt
import shutil
from phylib.io import model
import click
import re
import logging
import sys
import os
import time
import spikeglx
from one.alf import spec
import brainbox.metrics
import uuid
from ibldsp.voltage import decompress_destripe_cbin
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_log = logging.getLogger('SI-Kilosort4')
_log.setLevel(logging.INFO)


# May want to do a "remove duplicate spikes" after manual sorting  - this would allow manual sorting to merge  units that have some, but not a majority, of duplicated spikes

#TODO: Refactor into smaller functions?

# Parameters
job_kwargs = dict(chunk_duration="1s", n_jobs=15, progress_bar=True)
we_kwargs = dict()
sorter_params = dict(do_CAR=False,do_correction=True)
SI_MC = not sorter_params['do_correction']

# QC presets
AMPLITUDE_CUTOFF = 0.1
SLIDING_RP = 0.1
AMP_THRESH = 40.
MIN_SPIKES = 500

RUN_PC = False
MOTION_PRESET ='kilosort_like' # 'kilosort_like','nonrigid_accurate'
SORTER = 'kilosort4' 
SCRATCH_DIR = Path(r'D:/si_temp')

def print_elapsed_time(start_time):
    _log.info(f'Elapsed time: {time.time()-start_time:0.0f} seconds')

def move_motion_info(motion_path,destination):
    try:
        drift_depths = motion_path.joinpath('spatial_bins.npy')
        drift = motion_path.joinpath('motion.npy')
        drift_times = motion_path.joinpath('temporal_bins.npy')
        
        drift_depths.rename(destination.joinpath('drift_depths.um.npy'))
        drift.rename(destination.joinpath('drift.um.npy'))
        drift_times.rename(destination.joinpath('drift.times.npy'))
    except:
        _log.warning('SI computed motion not found')


def run_probe(probe_dir,probe_local,testing=False):
    start_time = time.time()

    # Set paths
    temp_local = probe_local.joinpath(f'.si_{uuid.uuid1()}')
    temp_local.mkdir(parents=True)
    PHY_DEST = probe_local.joinpath('kilosort4')
    # Temporary paths that will not be coming with us?
    SORT_PATH = temp_local.joinpath(f'.ks4')
    WVFM_PATH = temp_local.joinpath('.wvfm')
    PREPROC_PATH = temp_local.joinpath('.preproc')
    MOTION_PATH = temp_local.joinpath('.motion')
    probe_local.mkdir(parents=True,exist_ok=True)


    PREPROC_PATH.mkdir()
    if PHY_DEST.exists():
        _log.warning(f'Local phy destination exists ({PHY_DEST}). Skipping this probe {probe_dir}')
        return

    # POINT TO RECORDING and IBL Destripe 
    _log.info('Running destriping.')
    for ap_file in probe_dir.glob('*ap.bin'):
        destriped_bin = PREPROC_PATH.joinpath(ap_file.name)
        if destriped_bin.exists():
            _log.info(f'Destriped file {destriped_bin} exists. Skipping')
        shutil.copy(ap_file.with_suffix('.meta'),PREPROC_PATH)
        decompress_destripe_cbin(ap_file,output_file=destriped_bin) 
    stream = si.get_neo_streams('spikeglx',PREPROC_PATH)[0][0]
    recording = se.read_spikeglx(destriped_bin.parent,stream_id=stream)
    

    rec_list =[]
    for ii in range(recording.get_num_segments()):
        seg = recording.select_segments(ii)
        # THIS LINE CUTS THE SEGMENTS
        if testing:
            use_secs = 60
            _log.warning(f"TESTING: ONLY RUNNING ON {use_secs}s per segment")
            seg = seg.frame_slice(0,30000*use_secs)
        rec_list.append(seg)
    recording = si.concatenate_recordings(rec_list)
    recording.annotate(is_filtered=True)
    
    
    # RUN SORTER
    if SORT_PATH.exists():
        _log.info('='*100)
        _log.info('Found sorting. Loading...')
        sort_rez = si.load_extractor(SORT_PATH)
    else:
        sort_rez = ss.run_sorter_by_property(sorter_name=SORTER,
                                            recording=recording,
                                            grouping_property='group',
                                            working_folder=SORT_PATH.parent.joinpath('ks4_working'),
                                            verbose=True,
                                            remove_existing_folder=False,
                                            **sorter_params)
        sort_rez.save(folder=SORT_PATH)
    print_elapsed_time(start_time) 

    # WAVEFORMS
    if WVFM_PATH.exists():
        _log.info('Found waveforms. Loading')
        we = si.load_waveforms(WVFM_PATH)
        sort_rez = we.sorting
    else:
        _log.info('Extracting waveforms...')
        we = si.extract_waveforms(recording,sort_rez,folder=WVFM_PATH,sparse=False,**job_kwargs)
        _log.info('Removing redundant spikes...')
        sort_rez = si.remove_duplicated_spikes(sort_rez,censored_period_ms=0.166)
        sort_rez = si.remove_redundant_units(sort_rez,align=False,remove_strategy='max_spikes')
        we = si.extract_waveforms(recording,sort_rez,sparse=False,folder=WVFM_PATH,overwrite=True,**job_kwargs)

    _log.info('Comuting sparsity')
    sparsity = si.compute_sparsity(we,num_channels=9)
    print_elapsed_time(start_time) 

    # COMPUTE METRICS
    # Compute features
    _log.info('Computing amplitudes and locations')
    amplitudes = si.compute_spike_amplitudes(we,load_if_exists=True, **job_kwargs)
    locations = si.compute_spike_locations(we,load_if_exists=True,**job_kwargs)
    unit_locations = si.compute_unit_locations(we,load_if_exists=True)
    drift_metrics = si.compute_drift_metrics(we)

    if RUN_PC:
        pca = si.compute_principal_components(waveform_extractor=we, n_components=5, load_if_exists=True,mode='by_channel_local',**job_kwargs,sparsity=sparsity)
    

    # Compute metrics 
    _log.info('Computing metrics')
    metrics = si.compute_quality_metrics(waveform_extractor=we,load_if_exists=True)
    _log.info('Template metrics')
    template_metrics = si.compute_template_metrics(we,load_if_exists=True)

    # Perform automated quality control
    query = f'amplitude_cutoff<{AMPLITUDE_CUTOFF} & sliding_rp_violation<{SLIDING_RP} & amplitude_median>{AMP_THRESH}'
    good_unit_ids = metrics.query(query).index
    metrics['group'] = 'mua'
    metrics.loc[good_unit_ids,'group']='good'

    print_elapsed_time(start_time) 
    _log.info("Exporting to phy")
    si.export_to_phy(waveform_extractor=we,output_folder=PHY_DEST,
                    # sparsity=sparsity,
                    use_relative_path=True,copy_binary=True,
                    compute_pc_features=False,**job_kwargs)

    _log.info('Getting suggested merges')
    auto_merge_candidates = si.get_potential_auto_merge(we)
    pd.DataFrame(auto_merge_candidates).to_csv(PHY_DEST.joinpath('potential_merges.tsv'),sep='\t')
    
    # Make some appends to the phy destination
    spike_locations = np.vstack([locations['x'],locations['y']]).T
    np.save(PHY_DEST.joinpath('spike_locations.npy'),spike_locations)
    np.save(PHY_DEST.joinpath('cluster_locations.npy'),unit_locations)
    shutil.copy(PHY_DEST.joinpath('channel_groups.npy'),PHY_DEST.joinpath('cluster_shanks.npy'))

    for col in template_metrics:
        this_col = pd.DataFrame(template_metrics[col])
        this_col['cluster_id'] = sort_rez.get_unit_ids()
        this_col = this_col[['cluster_id',col]]
        this_col.to_csv(PHY_DEST.joinpath(f'cluster_{col}.tsv'),sep='\t',index=False)

    _log.info('Done sorting!')
    print_elapsed_time(start_time) 

    
    move_motion_info(MOTION_PATH,PHY_DEST)


@click.command()
@click.argument('session_path',type=click.Path())
@click.option('--dest','-d',default=None)
@click.option('--testing',is_flag=True)
@click.option('--no_move_final',is_flag=True)
@click.option('--no_clean',is_flag=True)
def run_session(session_path,dest,testing,no_move_final,no_clean):
    """Spike sort a session. A session is multiple simultanesouly recorded probes. Any instances of multiple 
    recordings must occur in the same anatomical location

    Args:
        session_path (_type_): Path to the session that contains folders for each probe.
        dest (_type_): Destination session path
        testing (_type_): Boolean flag to only run a small snippet of the data
        no_move_final (_type_): _description_
        no_clean (_type_): _description_
    """
    session_path = Path(session_path)
    rm_intermediate = True
    cleanup_local = not no_clean
    
    dest = dest or session_path
    dest = Path(dest).joinpath('alf')

    _log.debug(f'Destination set to {dest}')
    
    session_local = SCRATCH_DIR.joinpath(session_path.name)
    ephys_dir = session_path.joinpath('raw_ephys_data')
    _log.debug(f'{session_local=}')
    probe_dirs = list(ephys_dir.glob('probe*')) + list(ephys_dir.glob('*imec[0-9]'))
    

    # Loop over all the ephys files
    for probe_dir in probe_dirs:

        probe_local = session_local.joinpath(probe_dir.name)
        probe_dest = dest.joinpath(probe_dir.name)

        _log.info(
            '\n'+
            '='*100 +
            f'\nRunning SpikeInterface {SORTER}:' +
            f"\n\tGate: {session_path}" +
            f"\n\Probe: {probe_dir.name}\n" +
            '='*100
        )

        run_probe(probe_dir,probe_local,testing=testing)


        # Cleanup # TODO: Move out
        if rm_intermediate:
            try:
                shutil.rmtree(probe_local.joinpath('.si*'))
            except:
                _log.error('Could not delete temp si folder')

        if not no_move_final:
            if probe_dest.exists():
                _log.warning(f'Not moving because target {probe_dest} already exists')
                cleanup_local = False
            else:
                _log.info(f'Moving sorted data from {probe_local} to {dest}')
                shutil.move(str(probe_local),str(dest))


if __name__=='__main__':
    run_session()