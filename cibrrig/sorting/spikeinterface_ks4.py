'''
Run Kilosort 4 locally on the NPX computer. 
Data must be reorganized using the preprocess.ephys_data_to_alf.py script or it will not work
'''
import spikeinterface.extractors as se 
import spikeinterface.preprocessing as spre 
import spikeinterface.sorters as ss 
import spikeinterface.sortingcomponents.motion_interpolation as sim
import spikeinterface.full as si
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import click
import logging
import sys
import time
import one.alf.io as alfio
from ibllib.ephys.sync_probes import apply_sync
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_log = logging.getLogger('SI-Kilosort4')
_log.setLevel(logging.INFO)


# May want to do a "remove duplicate spikes" after manual sorting  - this would allow manual sorting to merge  units that have some, but not a majority, of duplicated spikes

# Parameters
job_kwargs = dict(chunk_duration="1s", n_jobs=15, progress_bar=True)
we_kwargs = dict()
sorter_params = dict(do_CAR=False,do_correction=True)
USE_MOTION_SI = not sorter_params['do_correction']
COMPUTE_MOTION_SI = True


# QC presets
AMPLITUDE_CUTOFF = 0.1
SLIDING_RP = 0.1
AMP_THRESH = 40.
MIN_SPIKES = 500

RUN_PC = False
MOTION_PRESET ='kilosort_like' # 'kilosort_like','nonrigid_accurate'
SORTER = 'kilosort4' 
SCRATCH_DIR = Path(r'D:/si_temp')
SCRATCH_DIR.mkdir(exist_ok=True)

def print_elapsed_time(start_time):
    _log.info(f'Elapsed time: {time.time()-start_time:0.0f} seconds')


def move_motion_info(motion_path,destination):
    """    
    Rename the motion data computed by Spikeinterface into a alf-like format
    If it doesn't exist, do nothing

    Args:
        motion_path (Path): Path to where the motion data live
        destination (_type_): Path to where the motion data land
    """    
    try:
        drift_depths = motion_path.joinpath('spatial_bins.npy')
        drift = motion_path.joinpath('motion.npy')
        drift_times = motion_path.joinpath('temporal_bins.npy')
        drift_fig = motion_path.joinpath('driftmap.png')
        drift_fig_zoom = motion_path.joinpath('driftmap_zoom.png')
        
        drift_depths.rename(destination.joinpath('drift_depths.um.npy'))
        drift.rename(destination.joinpath('drift.um.npy'))
        drift_times.rename(destination.joinpath('drift.times.npy'))
        drift_fig.rename(destination.joinpath('driftmap.png'))
        drift_fig_zoom.rename(destination.joinpath('driftmap_zoom.png'))
    except:
        _log.warning('SI computed motion not found')


def remove_opto_artifacts(recording, session_path, probe_path, object='laser',ms_before =0.5,ms_after=2.0):
    """Use the Spikeinterface "remove_artifacts" to zero out around the onsets and offsets of the laser
    Assumes an ALF format and existence of laser tables and sync objects

    Args:
        recording (spikeinterface recording extractor): 
        session_path (Path): 
        probe_dir (Path): 
        object (str, optional): ALF object. Defaults to 'laser'.
        ms_before (float, optional): Time before laser to blank. Defaults to 0.5.
        ms_after (float, optional): Time after laser to blank. Defaults to 2.0.
    """    
    rec_list = []
    for ii in range(recording.get_num_segments()):
        sync_fn = alfio.filter_by(probe_path,object=f'ephysData_*t{ii}',extra='sync')[0][0]
        segment = recording.select_segments(ii)
        _log.debug(segment.__repr__())
        opto_stims = alfio.load_object(session_path.joinpath('alf'),
                                    object=object,
                                    namespace='cibrrig',
                                    extra=f't{ii:.0f}',
                                    short_keys=True)
        opto_times = opto_stims.intervals.ravel()
        if len(opto_times)>0:
            opto_times_adj = apply_sync(probe_path.joinpath(sync_fn),opto_times,forward=False)
            opto_samps = np.array([recording.time_to_sample_index(x,segment_index=ii) for x in opto_times_adj])
            new_segment = si.remove_artifacts(segment,opto_samps,ms_before =ms_before,ms_after=ms_after)
            rec_list.append(new_segment)
        else:
            rec_list.append(segment)
    recording_out = si.append_recordings(rec_list)
    return(recording_out)


def concatenate_recording(recording,t0=0,tf=None):
    """
    Take a recording that may have multiple segments and turn it into a concatenated recording
    that Kilosort can handle.
    
    Optionally clip the dataset in time

    Args:
        recording (SI recording): May have multiple segments
        t0 (int, optional): start time in seconds. Defaults to 0.
        tf (_type_, optional): End time in seconds. Defaults to None - the entire recoridng
    """    
    rec_list =[]
    for ii in range(recording.get_num_segments()):
        seg = recording.select_segments(ii)
        if tf is not None:
            _log.warning(f"TESTING: ONLY RUNNING ON {tf-t0}s per segment")
            seg = seg.frame_slice(t0,30000*tf)
        rec_list.append(seg)
    recording = si.concatenate_recordings(rec_list)
    return(recording)


def si_motion(recording,MOTION_PATH):
    """
    Compute motion using the SI 

    Args:
        recording (SI recording):
        MOTION_PATH (Path): Path to save motion info to
    """    
    # Motion estimation
    if MOTION_PATH.exists():
        _log.info('Motion info loaded')
        motion_info = si.load_motion_info(MOTION_PATH)
        rec_mc = sim.interpolate_motion(recording = recording,
                                                    motion=motion_info['motion'],
                                                    temporal_bins=motion_info['temporal_bins'],
                                                    spatial_bins=motion_info['spatial_bins'],
                                                    **motion_info['parameters']['interpolate_motion_kwargs'],
                                                    )
    else:
        _log.info('Motion correction KS-like...')
        rec_mc, motion_info = si.correct_motion(recording, preset=MOTION_PRESET,
                        folder=MOTION_PATH,
                        output_motion_info=True, **job_kwargs)
    return(rec_mc,motion_info)


def plot_motion(motion_path):
    """
    Plot the motion and safe the figure

    Args:
        MOTION_PATH (Path): Directory where the motion info lives
        probe_local (Path): Directory to save the motion figures
    """    
    _log.info('Plotting motion info')
    try:
        motion_info = si.load_motion_info(motion_path)
        if not motion_path.joinpath('driftmap.png').exists():
            fig = plt.figure(figsize=(14, 8))
            si.plot_motion(motion_info, figure=fig, 
                        color_amplitude=True, amplitude_cmap='inferno', scatter_decimate=10)
            plt.savefig(motion_path.joinpath('driftmap.png'),dpi=300)
            for ax in fig.axes[:-1]:
                ax.set_xlim(30,60)
            plt.savefig(motion_path.joinpath('driftmap_zoom.png'),dpi=300)
    except:
        _log.error('Plotting motion failed')


def split_shanks_and_spatial_filter(rec):     
    """
    Split a multishank recording into multiple groups and perform spatial filtering

    Args:
        rec (_type_): _description_
    """       
    rec_split = rec.split_by(property='group')
    n_shanks = len(rec_split)
    _log.info(f'Found {n_shanks} channel groups')
    preprocessed_recordings = []
    for chan_group_rec in rec_split.values():
        rec_destriped = spre.highpass_spatial_filter(chan_group_rec)    
        preprocessed_recordings.append(rec_destriped)
    combined_preprocessed_recording = si.aggregate_channels(preprocessed_recordings)
    return(combined_preprocessed_recording)


def apply_preprocessing(recording,session_path,probe_dir,testing,skip_remove_opto = False):
    """
    Apply the IBL preprocessing
    Destripe, remove opto artifacts, concatenate

    Args:
        recording (_type_): _description_
        session_path (_type_): _description_
        probe_dir (_type_): _description_
        testing (_type_): _description_
    """    
    _log.info('Preprocessing IBL destripe...')
    rec_filtered = spre.highpass_filter(recording)
    rec_shifted = spre.phase_shift(rec_filtered)
    bad_channel_ids, all_channels = spre.detect_bad_channels(rec_shifted)
    rec_interpolated = spre.interpolate_bad_channels(rec_shifted, bad_channel_ids)
    rec_destriped = split_shanks_and_spatial_filter(rec_interpolated)

    if testing:
        rec_processed = rec_destriped
        _log.info("Testing, not removing opto artifacts")
    else:
        if not skip_remove_opto:
            rec_processed = remove_opto_artifacts(rec_destriped,session_path,probe_dir,ms_before=0.5,ms_after=2)
        else:
            rec_processed = rec_destriped
    
    tf = 60 if testing else None
    rec_out = concatenate_recording(rec_processed,tf=tf)
    return(rec_out)


def run_probe(probe_dir,probe_local,label='kilosort4',testing=False,skip_remove_opto=False):
    """Run spikesorting on a given probe

    Args:
        probe_dir (_type_): _description_
        probe_local (_type_): _description_
        label (str, optional): waht to label the phy output folder. Lives in alf/probeXX/. Defaults to 'kilosort4'.
        testing (bool, optional): Runs only on a subset of data. Defaults to False.
    """    
    start_time = time.time()

    # Set paths
    temp_local = probe_local.joinpath(f'.si')
    PHY_DEST = probe_local.joinpath(label)
    # Temporary paths that will not be coming with us?
    SORT_PATH = temp_local.joinpath(f'.ks4')
    WVFM_PATH = temp_local.joinpath('.wvfm')
    PREPROC_PATH = temp_local.joinpath('.preproc')
    MOTION_PATH = temp_local.joinpath('.motion')
    probe_local.mkdir(parents=True,exist_ok=True)


    if PHY_DEST.exists():
        _log.warning(f'Local phy destination exists ({PHY_DEST}). Skipping this probe {probe_dir}')
        return
    
    if not PREPROC_PATH.exists():
        stream = si.get_neo_streams('spikeglx',probe_dir)[0][0]
        recording = se.read_spikeglx(probe_dir,stream_id=stream)
        session_path = probe_dir.parent.parent

        # =========== Preprocessing =================== #
        rec_destriped = apply_preprocessing(recording,session_path,probe_dir,testing,skip_remove_opto=skip_remove_opto)

        # =============== Compute motion if requested.  ============ #
        if COMPUTE_MOTION_SI:
            rec_mc,motion = si_motion(rec_destriped,MOTION_PATH)

        # ============== Save motion if requested ============== #
        if COMPUTE_MOTION_SI and USE_MOTION_SI:
            final_preproc = rec_mc
        else:
            final_preproc = rec_destriped

        final_preproc.save(folder=PREPROC_PATH,**job_kwargs) 
        print_elapsed_time(start_time) 
    else:
        _log.info('Preprocessed data exists. Loading')

    # ============= Load preprocessed data from disk ============ # 
    recording = si.load_extractor(PREPROC_PATH)
    recording.annotate(is_filtered=True)

    # ============= RUN SORTER ==================== #
    if SORT_PATH.exists():
        _log.info('Found sorting. Loading...')
        sort_rez = si.load_extractor(SORT_PATH)
    else:
        _log.info(f"Running {SORTER}")
        sort_rez = ss.run_sorter_by_property(sorter_name=SORTER,
                                            recording=recording,
                                            grouping_property='group',
                                            working_folder=SORT_PATH.parent.joinpath('ks4_working'),
                                            verbose=True,
                                            remove_existing_folder=False,
                                            **sorter_params)
        sort_rez.save(folder=SORT_PATH)

    print_elapsed_time(start_time) 

    # ========= WAVEFORMS ============= #
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

    # ============ COMPUTE METRICS ============== #
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

    # =================== EXPORT ========= #
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
    shutil.copy(PHY_DEST.joinpath('channel_groups.npy'),PHY_DEST.joinpath('channel_shanks.npy'))

    for col in template_metrics:
        this_col = pd.DataFrame(template_metrics[col])
        this_col['cluster_id'] = sort_rez.get_unit_ids()
        this_col = this_col[['cluster_id',col]]
        this_col.to_csv(PHY_DEST.joinpath(f'cluster_{col}.tsv'),sep='\t',index=False)

    _log.info('Done sorting!')
    
    plot_motion(MOTION_PATH)
    move_motion_info(MOTION_PATH,PHY_DEST)
    return(PHY_DEST)



@click.command()
@click.argument('session_path',type=click.Path())
@click.option('--dest','-d',default=None)
@click.option('--testing',is_flag=True)
@click.option('--no_move_final',is_flag=True)
@click.option('--skip_remove_opto',is_flag=True,help='Flag to skip removal of the light artifacts. Probably advisable if light is presented far from the probe.')
def cli(session_path,dest,testing,no_move_final,skip_remove_opto):
    run(session_path,dest,testing,no_move_final,skip_remove_opto)

def run(session_path,dest=None,testing=False,no_move_final=False,skip_remove_opto=False):
    """Spike sort a session. A session is multiple simultanesouly recorded probes. Any instances of multiple 
    recordings must occur in the same anatomical location

    Args:
        session_path (_type_): Path to the session that contains folders for each probe.
        dest (_type_): Destination session path
        testing (_type_): Boolean flag to only run a small snippet of the data
        no_move_final (_type_): _description_
        no_clean (_type_): _description_
    """
    rm_intermediate = True
    move_final = not no_move_final
    label = 'kilosort4'

    # Get paths
    session_path = Path(session_path)
    session_local = SCRATCH_DIR.joinpath(session_path.parent.name+'_'+session_path.name)
    ephys_dir = session_path.joinpath('raw_ephys_data')
    _log.debug(f'{session_local=}')
    probe_dirs = list(ephys_dir.glob('probe*')) + list(ephys_dir.glob('*imec[0-9]'))

    
    # Set destination
    dest = dest or session_path
    dest = Path(dest).joinpath('alf')
    _log.debug(f'Destination set to {dest}')
    
    

    # ======= Loop over all probes in the session ========= #
    for probe_dir in probe_dirs:

        probe_local = session_local.joinpath(probe_dir.name)
        probe_dest = dest.joinpath(probe_dir.name)
        phy_dest = probe_dest.joinpath(label)
        phy_local = probe_local.joinpath(label)
        if phy_local.exists():
            _log.critical(f'Local PHY folder: {phy_local} exists. Not overwriting.')
            continue
        if phy_dest.exists():
            _log.critical(f'Destination PHY folder: {phy_dest} exists. Not overwriting.')
            continue

        _log.info(
            '\n'+
            '='*100 +
            f'\nRunning SpikeInterface {SORTER}:' +
            f"\n\tGate: {session_path}" +
            f"\n\tProbe: {probe_dir.name}\n" +
            '='*100
        )

        run_probe(probe_dir,probe_local,testing=testing,label=label,skip_remove_opto=skip_remove_opto)

        # ======= Remove temporary SI folder ========= #
        if rm_intermediate:
            try:
                shutil.rmtree(probe_local.joinpath('.si'))
            except:
                _log.error('Could not delete temp si folder')
        # ======= Move to destination ========= #
        if move_final:
            phy_dest = probe_dest.joinpath(phy_local.name)

            if phy_dest.exists():
                _log.warning(f'Not moving because target {phy_dest} already exists')
            else:
                _log.info(f'Moving sorted data from {phy_local} to {phy_dest}')
                shutil.move(str(phy_local),str(phy_dest))
    if move_final and rm_intermediate:
        _log.info(f'Removing {session_local}')
        shutil.rmtree(session_local)



if __name__=='__main__':
    cli()