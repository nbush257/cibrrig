"""
Run Kilosort 4 locally on the NPX computer.
Data must be reorganized using the preprocess.ephys_data_to_alf.py script first.
"""

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.sortingcomponents.motion.motion_interpolation as sim
import spikeinterface.full as si
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import shutil
import click
import logging
import sys
import time
import one.alf.io as alfio
from ibllib.ephys.sync_probes import apply_sync
from cibrrig.sorting.export_to_alf import ALFExporter, test_unit_refine_model_import


if sys.platform == "linux":
    import joblib

    N_JOBS = joblib.effective_n_jobs()
    CHUNK_DUR = "1s"
else:
    N_JOBS = 12
    CHUNK_DUR = "1s"

MOTION_PRESET = "dredge"  # 'kilosort_like','dredge'
SCRATCH_NAME = f"SCRATCH_{MOTION_PRESET}"

job_kwargs = dict(chunk_duration=CHUNK_DUR, n_jobs=N_JOBS, progress_bar=True)
si.set_global_job_kwargs(**job_kwargs)
do_correction = False
if MOTION_PRESET == "kilosort_like":
    do_correction = True
USE_MOTION_SI = not do_correction
sorter_params = dict(do_CAR=False, do_correction=do_correction)
COMPUTE_MOTION_SI = True
OPTO_OBJECTS = [
    "laser",
    "laser2",
]  # Alf objects to look for by default that we wish to remove the artifact for

EXTENSIONS = dict(
    random_spikes={"method": "uniform", "max_spikes_per_unit": 500, "seed": 42},
    waveforms={"ms_before": 1.3, "ms_after": 2.6},
    templates={"operators": ["average", "median", "std"]},
    noise_levels={},
    # amplitude_scalings = {},
    spike_amplitudes={},
    isi_histograms={},
    spike_locations={},
    unit_locations={},
    template_metrics={"include_multi_channel_metrics": True},
    correlograms={},
    template_similarity={},
)
np.random.seed(42)
# Set up logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


# Flags
SORTER = "kilosort4"


def log_elapsed_time(start_time):
    _log.info(f"Elapsed time: {time.time() - start_time:0.0f} seconds")


# TODO: test
def move_motion_info(src, destination):
    """
    Rename the motion data computed by Spikeinterface into a alf-like format
    If it doesn't exist, do nothing

    Args:
        motion_path (Path): Path to where the motion data live
        destination (Path): Path to where the motion data land
    """
    motion_dir = src.joinpath("motion")
    try:
        drift_depths = motion_dir.joinpath("spatial_bins_um.npy")
        drift = motion_dir.joinpath("displacement_seg0.npy")
        drift_times = motion_dir.joinpath("temporal_bins_s_seg0.npy")
        drift_fig = src.joinpath("driftmap.png")
        drift_fig_zoom = src.joinpath("driftmap_zoom.png")

        shutil.copy(drift_depths, destination.joinpath("drift_depths.um.npy"))
        shutil.copy(drift, destination.joinpath("drift.um.npy"))
        shutil.copy(drift_times, destination.joinpath("drift.times.npy"))
        shutil.copy(drift_fig, destination.joinpath("driftmap.png"))
        shutil.copy(drift_fig_zoom, destination.joinpath("driftmap_zoom.png"))
    except Exception:
        _log.warning("SI computed motion not found")


def remove_opto_artifacts(
    recording,
    session_path,
    probe_path,
    opto_objects=None,
    ms_before=0.125,
    ms_after=0.25,
):
    """
    Use the Spikeinterface "remove_artifacts" to zero out around the onsets and offsets of the laser
    Assumes an ALF format and existence of laser tables and sync objects

    Args:
        recording (spikeinterface recording extractor):
        session_path (Path):
        probe_dir (Path):
        object (str, optional): ALF object. Defaults to 'laser'.
        ms_before (float, optional): Time before laser to blank. Defaults to 0.125.
        ms_after (float, optional): Time after laser to blank. Defaults to 0.25.

    Returns:
        spikeinterface.RecordingExtractor: Recording extractor with artifacts removed.
    """

    def _align_artifacts(recording, samps, winsize=0.001):
        """
        Align artifact removal window to the peak of the artifact. Do this because
        the time stamp of the laser onset is not always the peak of the artifact.
        This allows us to cut a smaller chunk of data out

        Args:
            recording (spikeinterface.RecordingExtractor): Recording extractor object.
            samps (np.array): Array of sample indices.
            winsize (float, optional): Window size in seconds. Defaults to 0.001.

        Returns:
            np.array: Array of aligned sample indices.

        """
        samps_aligned = np.empty_like(samps)
        win_samps = int(winsize * recording.get_sampling_frequency())
        for ii, stim in enumerate(samps):
            _snippet = recording.frame_slice(
                stim - win_samps, stim + win_samps
            ).get_traces()
            samps_aligned[ii] = np.argmax(np.mean(_snippet**2, 1)) + stim - win_samps
        return samps_aligned

    # Set which alf objects to look for
    if opto_objects is None:
        opto_objects = OPTO_OBJECTS
    if not isinstance(opto_objects, list):
        opto_objects = [opto_objects]

    rec_list = []
    _log.info("Removing opto artifacts")
    for ii in range(recording.get_num_segments()):
        sync_fn = alfio.filter_by(probe_path, object=f"ephysData_*t{ii}", extra="sync")[
            0
        ][0]
        segment = recording.select_segments(ii)
        _log.debug(segment.__repr__())
        all_opto_times = []
        alf_path = session_path.joinpath("alf")
        for obj in opto_objects:
            if not alfio.exists(alf_path, obj):
                continue
            opto_stims = alfio.load_object(
                alf_path,
                object=obj,
                namespace="cibrrig",
                extra=f"t{ii:.0f}",
                short_keys=True,
            )
            opto_times = opto_stims.intervals.ravel()
            all_opto_times.append(opto_times)
        all_opto_times = np.sort(np.concatenate(all_opto_times))

        if len(all_opto_times) > 0:
            opto_times_adj = apply_sync(
                probe_path.joinpath(sync_fn), all_opto_times, forward=False
            )

            # Map times to samples in ints and align to peak artifact
            opto_samps = opto_times_adj * recording.get_sampling_frequency()
            opto_samps = np.round(opto_samps).astype(int)
            opto_samps = _align_artifacts(segment, opto_samps)

            # Blank out the artifacts
            new_segment = si.remove_artifacts(
                segment, opto_samps, ms_before=ms_before, ms_after=ms_after
            )

            rec_list.append(new_segment)
        else:
            rec_list.append(segment)
    recording_out = si.append_recordings(rec_list)
    return recording_out


def concatenate_recording(recording, t0=0, tf=None):
    """
    Concatenate a multi-segment recording into a single continuous recording.

    This function takes a recording that may have multiple segments and concatenates them
    into a single continuous recording that Kilosort can handle. Optionally, it can clip
    the dataset in time.

    Args:
        recording (spikeinterface.RecordingExtractor): Recording extractor object that may have multiple segments.
        t0 (int, optional): Start time in seconds. Defaults to 0.
        tf (int, optional): End time in seconds. Defaults to None, which means the entire recording.

    Returns:
        spikeinterface.RecordingExtractor: Concatenated recording extractor object.
    """

    rec_list = []
    for ii in range(recording.get_num_segments()):
        seg = recording.select_segments(ii)
        if tf is not None:
            _log.warning(f"TESTING: ONLY RUNNING ON {tf - t0}s per segment")
            sf = int(seg.get_sampling_frequency() * tf)
            sf = np.min([sf, seg.get_num_frames()])
            s0 = int(seg.get_sampling_frequency() * t0)
            s0 = np.max([s0, 0])
            seg = seg.frame_slice(s0, sf)
        rec_list.append(seg)
    recording = si.concatenate_recordings(rec_list)
    return recording


def si_motion(recording, MOTION_PATH):
    """
    Compute motion using SpikeInterface (SI) and save the motion information.

    This function estimates the motion of the recording using SpikeInterface. If motion information
    already exists at the specified path, it loads the existing motion information and interpolates
    the motion. Otherwise, it performs motion correction and saves the motion information.

    Args:
        recording (spikeinterface.RecordingExtractor): Recording extractor object.
        MOTION_PATH (Path): Path to save or load motion information.

    Returns:
        tuple: A tuple containing:
            - spikeinterface.RecordingExtractor: Motion-corrected recording extractor object.
            - dict: Motion information dictionary.
    """

    # Motion estimation
    if MOTION_PATH.exists():
        _log.info("Motion info loaded")
        motion_info = si.load_motion_info(MOTION_PATH)
        rec_mc = sim.interpolate_motion(
            recording=recording, motion=motion_info["motion"]
        )
    else:
        _log.info(f"Motion correction {MOTION_PRESET}...")
        rec_mc, motion_info = si.correct_motion(
            recording,
            preset=MOTION_PRESET,
            folder=MOTION_PATH,
            output_motion_info=True,
            **job_kwargs,
        )
    return (rec_mc, motion_info)


def plot_motion(motion_path, rec):
    """
    Plot the motion information and save the figure.

    This function loads the motion information from the specified path, plots the motion,
    and saves the figure as 'driftmap.png' and 'driftmap_zoom.png'.

    Args:
        MOTION_PATH (Path): Directory where the motion info lives
        rec (SI recording): Recording to plot the motion on
    """
    _log.info("Plotting motion info")
    try:
        motion_info = si.load_motion_info(motion_path)
        if not motion_path.joinpath("driftmap.png").exists():
            fig = plt.figure(figsize=(14, 8))
            si.plot_motion_info(
                motion_info,
                rec,
                figure=fig,
                color_amplitude=True,
                amplitude_cmap="inferno",
                scatter_decimate=10,
            )
            plt.savefig(motion_path.joinpath("driftmap.png"), dpi=300)
            for ax in fig.axes[:-1]:
                ax.set_xlim(30, 60)
            plt.savefig(motion_path.joinpath("driftmap_zoom.png"), dpi=300)
    except Exception as e:
        _log.error("Plotting motion failed")
        _log.error(e)


def split_shanks_and_spatial_filter(rec):
    """
    Split a multishank recording into multiple groups and perform spatial filtering.

    This function splits a multishank recording into separate channel groups based on the 'group' property.
    It then applies a highpass spatial filter to each channel group and combines the preprocessed recordings
    into a single recording.

    Args:
        rec (spikeinterface.RecordingExtractor): Recording extractor object containing the multishank recording.

    Returns:
        spikeinterface.RecordingExtractor: Combined recording extractor object with spatially filtered data.
    """

    # Split the recording into separate channel groups based on the 'group' property
    rec_split = rec.split_by(property="group")
    n_shanks = len(rec_split)
    _log.info(f"Found {n_shanks} channel groups")

    preprocessed_recordings = []
    for chan_group_rec in rec_split.values():
        # Apply highpass spatial filter to each channel group
        rec_destriped = spre.highpass_spatial_filter(chan_group_rec)
        preprocessed_recordings.append(rec_destriped)

    # Combine the preprocessed recordings into a single recording
    combined_preprocessed_recording = si.aggregate_channels(preprocessed_recordings)
    return combined_preprocessed_recording


def remove_and_interpolate(
    recording, probe_dir, t0=0, tf=120, remove=True, plot=True, save=True
):
    """Remove channels outside the brain and interpolate bad channels

    Args:
        recording (spikeinterface.RecordingExtractor): Recording extractor object.
        t0 (float, optional): Start time in seconds. Defaults to 0.
        tf (float, optional): End time in seconds. Defaults to 120.
        remove (bool, optional): If True, remove channels outside the brain. Defaults to True.
        plot (bool, optional): If True, plot the traces before and after removing bad channels. Defaults to True.
        save (bool, optional): If True, save the channel labels. Defaults to True.

    Returns:
        spikeinterface.RecordingExtractor: Recording extractor object with bad channels removed and interpolated.
        np.array: Array of channel indices that were removed.
    """
    _log.info("Removing and interpolating bad channels")

    # Map times to samples with recording start as t0 (fix since recording start is not always 0 in spikeinterface>0.100ish)
    sr = recording.get_sampling_frequency()
    s0, sf = sr * t0, sr * tf
    s0 = np.round(s0).astype(int)
    sf = np.round(sf).astype(int)

    # Get the segment between t0 and tf by indexing into frames
    recording_sub = si.select_segment_recording(recording, 0)  # Grab the first segment
    s0 = np.max([s0, 0])
    sf = np.min([sf, recording_sub.get_num_frames()])
    recording_sub = recording_sub.frame_slice(s0, sf)

    # Detect bad channels
    _, chan_labels = si.detect_bad_channels(
        recording_sub, outside_channels_location="both"
    )

    out_channels = np.where(chan_labels == "out")[0]

    # Set dead or noise channels to bad (i.e., exclude out channels)
    bad_channels = recording.channel_ids[np.isin(chan_labels, ["dead", "noise"])]

    # Remove channels outside the brain
    if remove:
        recording_good = recording.remove_channels(recording.channel_ids[out_channels])
        recording_good = si.interpolate_bad_channels(recording_good, bad_channels)
    else:
        recording_good = si.interpolate_bad_channels(recording, bad_channels)

    if plot:
        f, ax = plt.subplots(ncols=3, sharey=True)
        t0 = recording.get_start_time() + 10
        tf = t0 + 4
        tf = min(tf, recording.get_end_time())

        si.plot_traces(
            recording, time_range=(t0, tf), clim=(-50, 50), ax=ax[0], segment_index=0
        )
        si.plot_traces(
            recording_good,
            time_range=(t0, tf),
            clim=(-50, 50),
            ax=ax[1],
            segment_index=0,
        )

        ax[2].plot(chan_labels, recording.get_channel_locations()[:, 1])
        ax[0].set_title("Original")
        ax[1].set_title("Removed and interpolated")
        ax[0].set_ylim(0, 3840)
        if save:
            plt.savefig(probe_dir.joinpath("remove_and_interpolate.png"), dpi=300)
        plt.close("all")
    if save:
        np.save(
            probe_dir.joinpath("_spikeinterface_ephysChannels.siLabels.npy"),
            chan_labels,
        )

    return (recording_good, chan_labels)


def apply_preprocessing(
    recording, session_path, probe_dir, testing, skip_remove_opto=False
):
    """
    Apply the IBL preprocessing pipeline to the recording.

    This function applies a series of preprocessing steps to the recording
    1. Highpass filtering
    2. Phase shifting
    3. Bad channel detection and interpolation
    4. Spatial filtering (destriping)

    Optionally, it can also remove optogenetic artifacts and concatenate recording segments.

    Args:
        recording (spikeinterface.RecordingExtractor): Recording extractor object.
        session_path (str or Path): Path to the session directory.
        probe_dir (str or Path): Path to the probe directory.
        testing (bool): If True, run in testing mode with limited data.
        skip_remove_opto (bool, optional): If True, skip the removal of optogenetic artifacts. Defaults to False.

    Returns:
        spikeinterface.RecordingExtractor: Preprocessed and concatenated recording extractor object.
    """
    _log.info("Preprocessing IBL destripe...")

    # Apply highpass filter to the recording
    rec_filtered = spre.highpass_filter(recording)

    # Apply phase shift to the filtered recording
    rec_shifted = spre.phase_shift(rec_filtered)

    # Remove channels outside the brain and interpolate bad channels
    rec_interpolated, chan_labels = remove_and_interpolate(
        rec_shifted, probe_dir, remove=True, plot=True, save=True
    )
    plt.close("all")

    # Apply spatial filtering and split shanks
    rec_destriped = split_shanks_and_spatial_filter(rec_interpolated)

    if testing:
        rec_processed = rec_destriped
        _log.info("Testing, not removing opto artifacts")
    else:
        if not skip_remove_opto:
            # Remove optogenetic artifacts if not skipped
            rec_processed = remove_opto_artifacts(
                rec_destriped, session_path, probe_dir
            )
        else:
            rec_processed = rec_destriped

    # Set the end time to 60 if testing and concatenate the recording
    tf = 60 if testing else None
    rec_out = concatenate_recording(rec_processed, tf=tf)
    return rec_out


# TODO:  test
def extract_breath_events(session_path, dest):
    """
    Create an 'events.csv' that has the times of each breath for the alf folder.
    This is then used in phy if the PSTH plugin exists.
    Args:
        session_path (Path): Path to the session directory where the original data exists
        alf_path (Path): Path to the ALF sorted data

    Returns:
        None

    """
    breaths_fn, attributes = alfio.filter_by(
        session_path.joinpath("alf"), object="breaths", attribute="times"
    )
    if len(breaths_fn) == 0:
        _log.info("No breath events found. Not exporting for phy curation")
        return
    elif len(breaths_fn) > 1:
        _log.error("Multiple breath events found. Not exporting for phy curation")
        return

    breaths = alfio.load_object(
        session_path.joinpath("alf"),
        object="breaths",
        attribute="times",
        short_keys=True,
    ).to_df()
    breaths.to_csv(dest.joinpath("events.csv"), index=False, header=False)


def postprocess_sorting(analyzer_path, recording, sort_rez):
    """
    Postprocess the sorting result. Saves raw and automerged versions of the sorting analyzer to disk as .zarr files.

    Performs these steps:
    1. Creates a sorting analyzer in memory
    2. Compute all extensions requested in global EXTENSIONS
    3. Remove redundant units
    4. Compute PCs
    5. Compute quality metrics
    6. Saves the analyzer (as .raw.zarr)
    7. Auto-merge units
    8. Recompute metrics on merged data (required to prevent crashes downstream)
    9. Save the automerged analyzer (as .zarr)


    Args:
        analyzer_path (Path): Path to save the analyzer.
        recording (spikeinterface.RecordingExtractor): Recording extractor object.
        sort_rez (spikeinterface.SortingExtractor): Sorting extractor object.

    Returns:
        spikeinterface.SortingAnalyzer: Postprocessed sorting analyzer object.
    """
    n_pca_jobs = N_JOBS if sys.platform == "linux" else 1
    # Create analyzer
    analyzer = si.create_sorting_analyzer(
        sorting=sort_rez,
        recording=recording,
        num_channels=12,
        method="closest_channels",
    )

    # Compute extensions
    analyzer.compute_several_extensions(EXTENSIONS)

    # Remove redundant units
    clean_sort_rez = si.remove_redundant_units(analyzer)
    analyzer = analyzer.select_units(clean_sort_rez.unit_ids)

    # Compute PCs

    analyzer.compute(
        "principal_components",
        n_components=3,
        mode="by_channel_local",
        n_jobs=n_pca_jobs,
    )
    analyzer.compute("quality_metrics")

    # Stash the pre-merged analyzer
    analyzer.save_as(folder=analyzer_path.with_suffix(".raw.zarr"), format="zarr")

    # Auto_merge units
    analyzer = si.auto_merge_units(
        sorting_analyzer=analyzer,
        presets=["temporal_splits", "similarity_correlograms"],
        censor_ms=0.166,
        recursive=True,
    )

    # Recompute metrics on merged data to allow for autolabel
    analyzer.compute_several_extensions(EXTENSIONS)
    analyzer.compute(
        "principal_components",
        n_components=3,
        mode="by_channel_local",
        n_jobs=n_pca_jobs,
    )
    analyzer.compute("quality_metrics")

    # Save the automerged analyzer
    analyzer.save_as(folder=analyzer_path.with_suffix(".zarr"), format="zarr")
    return analyzer


def move_sorted_to_alf(sorted_dir, probe_local):
    """
    Move the sorted data from the local scratch directory to the alf directory.

    Args:
        sorted_dir (Path): Path to the sorted data directory.
        probe_local (Path): Path to the probe directory in the alf folder.
    Returns:
        None
    """
    for item in sorted_dir.iterdir():
        dest = probe_local.joinpath(item.name)
        if item.is_dir():
            shutil.move(str(item), str(dest))
        else:
            shutil.move(str(item), str(dest))
    shutil.rmtree(sorted_dir)


def run_probe(probe_src, probe_local, testing=False, skip_remove_opto=False):
    """
    Run spikesorting on a single probe

    Args:
        probe_dir (Path): Path to the probe directory.
        probe_local (str): Local path to save phy sorting to.
        testing (bool, optional): If True, run in testing mode (short data snippet). Defaults to False.
        skip_remove_opto (bool, optional): If True, skip the removal of opto artifacts. Defaults to False.

    Returns:
        Path: Path to the sorted data.
    """
    start_time = time.time()


    # Set paths
    si_path = probe_local.joinpath(".si")
    # Temporary paths that will not be coming with us?
    preproc_path = si_path.joinpath(".preprocessed")
    sort_path = si_path.joinpath(".sort")
    motion_path = si_path.joinpath(".motion")
    analyzer_path = si_path.joinpath(".analyzer")
    exported_alf_path = si_path.joinpath("kilosort4")
    probe_local.mkdir(parents=True, exist_ok=True)
    #
    stream = si.get_neo_streams("spikeglx", probe_src)[0][0]
    recording = se.read_spikeglx(probe_src, stream_id=stream)
    session_path = probe_src.parent.parent

    # =========== #
    # =========== Preprocessing =================== #
    # =========== #
    if not preproc_path.with_suffix(".zarr").exists():
        rec_destriped = apply_preprocessing(
            recording,
            session_path,
            probe_src,
            testing,
            skip_remove_opto=skip_remove_opto,
        )

        # =============== Compute motion if requested.  ============ #
        if COMPUTE_MOTION_SI:
            rec_mc, motion = si_motion(rec_destriped.astype("float32"), motion_path)

        # ============== Save motion if requested ============== #
        if COMPUTE_MOTION_SI and USE_MOTION_SI:
            recording = rec_mc
        else:
            recording = rec_destriped
        recording = recording.astype("int16")
        recording.save(folder=preproc_path, format="zarr")
        del recording
    _log.info("Loading preprocessed recording")
    recording = si.load(preproc_path.with_suffix(".zarr"))

    # ============= RUN SORTER ==================== #

    if sort_path.exists():
        _log.info("Found sorting. Loading...")
        sort_rez = si.load(sort_path)
    else:
        _log.info(f"Running {SORTER}")
        # job_kwargs = dict(chunk_duration=CHUNK_DUR, n_jobs=1, progress_bar=True)
        # si.set_global_job_kwargs(**job_kwargs)
        sort_rez = ss.run_sorter(
            sorter_name=SORTER,
            recording=recording,
            folder=sort_path,
            verbose=True,
            remove_existing_folder=False,
            n_jobs=1,
            **sorter_params,
        )

        # Remove kilosort handler
        try:
            ks_log = logging.getLogger("kilosort")
            for h in ks_log.handlers:
                h.close()
                ks_log.removeHandler(h)
        except Exception as e:
            _log.error(f"Could not remove kilosort log handlers: {e}")
    sort_rez = si.remove_duplicated_spikes(
        sort_rez, method="keep_first_iterative", censored_period_ms=0.166
    )

    _log.info("Finished sorting:")
    log_elapsed_time(start_time)

    # Subset to a small number of units if testing
    if testing:
        unit_ids = sort_rez.get_unit_ids()
        keep_units = np.random.choice(
            unit_ids, size=min(40, len(unit_ids)), replace=False
        )
        sort_rez = sort_rez.select_units(keep_units)
        _log.info(f"Testing, only keeping {len(keep_units)} units")

    # ============= POSTPROCESSING ============= #
    _log.info("Computing waveforms and QC")
    if analyzer_path.with_suffix(".zarr").exists():
        _log.info("Found analyzer. Loading...")
        analyzer = si.load_sorting_analyzer(folder=analyzer_path.with_suffix(".zarr"))
    else:
        analyzer = postprocess_sorting(analyzer_path, recording, sort_rez)

    # ============= EXPORT ============= #
    _log.info("Exporting to ALF")
    exporter = ALFExporter(
        analyzer=analyzer,
        dest=exported_alf_path,
        bin_path=probe_src,
        job_kwargs=si.get_global_job_kwargs(),
        testing=testing,
    )
    exporter.run()

    extract_breath_events(session_path, exported_alf_path)

    # Copy motion info to alf folder
    _log.info("Copying motion info to alf folder")
    plot_motion(motion_path, recording)
    move_motion_info(motion_path, exported_alf_path)

    # ============= MOVE TO ALF ============= #
    _log.info("Moving sorted data to alf folder")
    move_sorted_to_alf(exported_alf_path, probe_local)


    del sort_rez
    del analyzer
    del recording


@click.command()
@click.argument("session_path", type=click.Path())
@click.option(
    "--dest",
    "-d",
    default=None,
    help="Destination folder for sorted data. Generates subfolders for each probe. Defaults to session/alf/<sorter>",
)
@click.option(
    "--testing",
    is_flag=True,
    help="Run in testing mode with reduced data for quick checks (60s segment).",
)
@click.option(
    "--no_move_final",
    is_flag=True,
    help="Prevent moving final output files to the destination directory. Copy stays in scratch.",
)
@click.option(
    "--keep_scratch",
    is_flag=True,
    help="Retain intermediate scratch files after processing. Overridden and set to false if not moving final.",
)
@click.option(
    "--skip_remove_opto",
    is_flag=True,
    help="Flag to skip removal of the light artifacts. Probably advisable if light is presented far from the probe.",
)
def cli(session_path, dest, testing, no_move_final, skip_remove_opto, keep_scratch):
    run(
        session_path,
        dest,
        testing,
        no_move_final,
        skip_remove_opto,
        rm_intermediate=not keep_scratch,
    )


def run(
    session_path,
    dest=None,
    testing=False,
    no_move_final=False,
    skip_remove_opto=False,
    rm_intermediate=True,
):
    """
    Spike sort a session. A session is multiple simultanesouly recorded probes. Any instances of multiple
    recordings must occur in the same anatomical location

    If a destination is not provided, the sorted data will be placed in the `session/alf/<sorter>` directory.
    Args:
        session_path (str or Path): Path to the session directory.
        dest (str or Path, optional): Destination directory for the sorted data. Defaults to None.
        testing (bool, optional): If True, run in testing mode. Defaults to False.
        no_move_final (bool, optional): If True, do not move the final sorted data. Defaults to False.
        skip_remove_opto (bool, optional): If True, skip the removal of opto artifacts. Defaults to False.
        rm_intermediate (bool, optional): If True, remove intermediate files. Defaults to True.
    """
    move_final = not no_move_final
    if not move_final:
        rm_intermediate = False

    # Get paths
    session_path = Path(session_path)
    if sys.platform == "linux":
        SCRATCH_DIR = session_path.joinpath(SCRATCH_NAME)
    else:
        if Path(r"D:/").exists():
            SCRATCH_DIR = Path("D:/").joinpath(SCRATCH_NAME)
        else:
            SCRATCH_DIR = Path("C:/").joinpath(SCRATCH_NAME)
    session_local = SCRATCH_DIR.joinpath(
        session_path.parent.name + "_" + session_path.name
    )

    test_unit_refine_model_import()

    ephys_dir = session_path.joinpath("raw_ephys_data")
    _log.debug(f"{session_local=}")
    probe_dirs = list(ephys_dir.glob("probe*")) + list(ephys_dir.glob("*imec[0-9]"))
    n_probes = len(probe_dirs)
    _log.info(f"{n_probes=}")

    # Set destination
    dest = dest or session_path.joinpath("alf")
    _log.debug(f"Destination set to {dest}")

    # ======= Loop over all probes in the session ========= #
    for probe_src in probe_dirs:
        probe_name = probe_src.name  # e.g.,probe00
        # Set up the paths
        probe_alf_local = session_local.joinpath(probe_src.name)
        probe_alf_remote = dest.joinpath(probe_src.name)

        has_local = probe_alf_local.joinpath("params.py").exists()
        has_remote = probe_alf_remote.joinpath("params.py").exists()

        if has_local:
            _log.critical(
                f"Sorted data found at {probe_alf_local}."
            )

        if has_remote:
            _log.critical(
                f"Sorted data found at {probe_alf_remote}."
            )
        if has_remote:
            _log.info(
                f"Skipping probe {probe_name} since sorted data already exists remotely"
            )
            continue
        # ======= Run the sorter ========= #
        if not has_local and not has_remote:
            _log.info(
                "\n"
                + "=" * 100
                + f"\nRunning SpikeInterface {SORTER}:"
                + f"\n\tSession: {session_path}"
                + f"\n\tProbe: {probe_src.name}"
                + f"\n\t{probe_alf_local = }"
                + f"\n\t{probe_alf_remote = }"
                + f"\n\t{testing = }"
                + f"\n\t{skip_remove_opto = }"
                + f"\n\t{USE_MOTION_SI = }"
                + f"\n\t{probe_name = }"
                + f"\n\t{N_JOBS = }"
                + f"\n\t{CHUNK_DUR = }\n"
                + "=" * 100
            )
            run_probe(
                probe_src,
                probe_alf_local,
                testing=testing,
                skip_remove_opto=skip_remove_opto,
            )

        # ======= Move to destination ========= #
        if move_final:
            if probe_alf_remote.exists():
                _log.warning(
                    f"Not moving because target {probe_alf_remote} already exists. Not deleting local copy {probe_alf_local}"
                )
                rm_intermediate = False
            else:
                _log.info(
                    f"Moving sorted data from {probe_alf_local} to {probe_alf_remote}"
                )
                shutil.move(str(probe_alf_local), str(probe_alf_remote))

    # ======= Remove temporary SI folder ========= #
    if rm_intermediate and n_probes > 0:
        shutil.rmtree(SCRATCH_DIR)


if __name__ == "__main__":
    cli()
