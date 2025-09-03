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
from cibrrig.sorting.export_to_alf import ALFExporter


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
    random_spikes={"method": "uniform", "max_spikes_per_unit": 600, "seed": 42},
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

# Set up logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("SI-Kilosort4")
_log.setLevel(logging.INFO)


# Flags
RUN_PC = False
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
    try:
        drift_depths = src.joinpath("spatial_bins.npy")
        drift = src.joinpath("motion.npy")
        drift_times = src.joinpath("temporal_bins.npy")
        drift_fig = src.joinpath("driftmap.png")
        drift_fig_zoom = src.joinpath("driftmap_zoom.png")

        drift_depths.rename(destination.joinpath("drift_depths.um.npy"))
        drift.rename(destination.joinpath("drift.um.npy"))
        drift_times.rename(destination.joinpath("drift.times.npy"))
        drift_fig.rename(destination.joinpath("driftmap.png"))
        drift_fig_zoom.rename(destination.joinpath("driftmap_zoom.png"))
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

    This function applies a series of preprocessing steps to the recording, including:
    highpass filtering,
    phase shifting
    bad channel detection and interpolation
    spatial filtering.
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
def extract_breath_events(session_path, alf_path):
    """
    Create an 'events.csv' that has the times of each breath for the alf folder.
    This is then used in phy if the PSTH plugin exists.
    Args:
        session_path (Path): Path to the session directory where the original data exists
        alf_path (Path): Path to the ALF sorted data

    Returns:
        None
    """
    if alfio.exists(alf_path, "breaths"):
        breath_times = alfio.load_object(
            session_path.joinpath("alf"), "breaths"
        ).to_df()["times"]
        breath_times.to_csv(alf_path.joinpath("events.csv"), index=False, header=False)
    else:
        _log.info("No breath events found. Not exporting for phy curation")


def run_probe(
    probe_dir, probe_local, label="kilosort4", testing=False, skip_remove_opto=False
):
    """
    Run spikesorting on a single probe

    Args:
        probe_dir (Path): Path to the probe directory.
        probe_local (str): Local path to save phy sorting to.
        label (str, optional): Label for the sorting. Defaults to 'kilosort4'.
        testing (bool, optional): If True, run in testing mode (short data snippet). Defaults to False.
        skip_remove_opto (bool, optional): If True, skip the removal of opto artifacts. Defaults to False.

    Returns:
        Path: Path to the sorted data.
    """
    start_time = time.time()

    # Set paths
    temp_local = probe_local.joinpath(".si")
    PHY_DEST = probe_local.joinpath(label)
    # Temporary paths that will not be coming with us?
    PREPROC_PATH = temp_local.joinpath(".preprocessed")
    SORT_PATH = temp_local.joinpath(".sort")
    MOTION_PATH = temp_local.joinpath(".motion")
    ANALYZER_PATH = temp_local.joinpath(".analyzer")
    probe_local.mkdir(parents=True, exist_ok=True)

    if PHY_DEST.exists():
        _log.warning(
            f"Local phy destination exists ({PHY_DEST}). Skipping this probe {probe_dir}"
        )
        return

    stream = si.get_neo_streams("spikeglx", probe_dir)[0][0]
    recording = se.read_spikeglx(probe_dir, stream_id=stream)
    session_path = probe_dir.parent.parent

    # =========== #
    # =========== Preprocessing =================== #
    # =========== #
    if not PREPROC_PATH.with_suffix(".zarr").exists():
        rec_destriped = apply_preprocessing(
            recording,
            session_path,
            probe_dir,
            testing,
            skip_remove_opto=skip_remove_opto,
        )

        # =============== Compute motion if requested.  ============ #
        if COMPUTE_MOTION_SI:
            rec_mc, motion = si_motion(rec_destriped.astype("float32"), MOTION_PATH)

        # ============== Save motion if requested ============== #
        if COMPUTE_MOTION_SI and USE_MOTION_SI:
            recording = rec_mc
        else:
            recording = rec_destriped
        recording = recording.astype("int16")
        recording.save(folder=PREPROC_PATH, format="zarr")
        del recording
    _log.info("Loading preprocessed recording")
    recording = si.load(PREPROC_PATH.with_suffix(".zarr"))

    job_kwargs = dict(chunk_duration=CHUNK_DUR, n_jobs=1, progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)
    # ============= RUN SORTER ==================== #
    if SORT_PATH.exists():
        _log.info("Found sorting. Loading...")
        sort_rez = si.read_sorter_folder(SORT_PATH)
    else:
        _log.info(f"Running {SORTER}")
        sort_rez = ss.run_sorter(
            sorter_name=SORTER,
            recording=recording,
            folder=SORT_PATH,
            verbose=True,
            remove_existing_folder=False,
            n_jobs=1,
            **sorter_params,
        )

    sort_rez = si.remove_duplicated_spikes(
        sort_rez, method="keep_first_iterative", censored_period_ms=0.166
    )
    job_kwargs = dict(chunk_duration=CHUNK_DUR, n_jobs=N_JOBS, progress_bar=True)
    si.set_global_job_kwargs(**job_kwargs)
    _log.info("Finished sorting:")
    log_elapsed_time(start_time)

    _log.info("Computing waveforms and QC")
    if ANALYZER_PATH.exists():
        analyzer = si.load_sorting_analyzer(folder=ANALYZER_PATH.with_suffix(".zarr"))
    else:
        analyzer = si.create_sorting_analyzer(
            sorting=sort_rez,
            recording=recording,
        )
        analyzer.compute_several_extensions(EXTENSIONS)

        # Remove redundant units
        clean_sort_rez = si.remove_redundant_units(analyzer)
        analyzer = analyzer.select_units(clean_sort_rez.unit_ids)

        # Compute PCs (Must be global for ALF conversion)
        analyzer.compute(
            "principal_components", n_components=3, mode="by_channel_local"
        )
        analyzer.compute("quality_metrics")

        # Stash the pre-merged analyzer
        analyzer.save_as(folder=ANALYZER_PATH.with_suffix(".raw.zarr"), format="zarr")

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
            "principal_components", n_components=3, mode="by_channel_local"
        )
        analyzer.compute("quality_metrics")

        # Save the automerged analyzer
        analyzer.save_as(folder=ANALYZER_PATH.with_suffix(".zarr"), format="zarr")

    # ============= EXPORT ============= #
    _log.info("Exporting to ALF")
    alf_path = PHY_DEST.parent.joinpath("small_alf")
    exporter = ALFExporter(analyzer, alf_path, job_kwargs=job_kwargs)
    exporter.run()

    # TODO: Sync
    extract_breath_events(session_path, alf_path)
    # TODO: Test
    plot_motion(MOTION_PATH, recording)
    shutil.move(str(MOTION_PATH), str(alf_path))


@click.command()
@click.argument("session_path", type=click.Path())
@click.option("--dest", "-d", default=None)
@click.option("--testing", is_flag=True)
@click.option("--no_move_final", is_flag=True)
@click.option("--keep_scratch", is_flag=True)
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
        rm_intermediate=~keep_scratch,
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
    label = SORTER

    # Get paths
    session_path = Path(session_path)
    if sys.platform == "linux":
        SCRATCH_DIR = session_path.joinpath(SCRATCH_NAME)
    else:
        SCRATCH_DIR = Path("D:/").joinpath(SCRATCH_NAME)
    session_local = SCRATCH_DIR.joinpath(
        session_path.parent.name + "_" + session_path.name
    )

    ephys_dir = session_path.joinpath("raw_ephys_data")
    _log.debug(f"{session_local=}")
    probe_dirs = list(ephys_dir.glob("probe*")) + list(ephys_dir.glob("*imec[0-9]"))
    n_probes = len(probe_dirs)
    _log.info(f"{n_probes=}")

    # Set destination
    dest = dest or session_path
    dest = Path(dest).joinpath("alf")
    _log.debug(f"Destination set to {dest}")

    # ======= Loop over all probes in the session ========= #
    for probe_dir in probe_dirs:
        # Set up the paths
        probe_local = session_local.joinpath(probe_dir.name)
        probe_dest = dest.joinpath(probe_dir.name)
        phy_dest = probe_dest.joinpath(label)
        phy_local = probe_local.joinpath(label)
        if phy_local.exists():
            _log.critical(f"Local PHY folder: {phy_local} exists. Not overwriting.")
            continue
        if phy_dest.exists():
            _log.critical(
                f"Destination PHY folder: {phy_dest} exists. Not overwriting."
            )
            continue
        # ======= Run the sorter ========= #
        _log.info(
            "\n"
            + "=" * 100
            + f"\nRunning SpikeInterface {SORTER}:"
            + f"\n\tGate: {session_path}"
            + f"\n\tProbe: {probe_dir.name}"
            + f"\n\t{dest=}"
            + f"\n\t{testing=}"
            + f"\n\t{skip_remove_opto=}"
            + f"\n\t{USE_MOTION_SI=}"
            + f"\n\t{label=}"
            + f"\n\t{N_JOBS=}"
            + f"\n\t{CHUNK_DUR=}\n"
            + "=" * 100
        )
        run_probe(
            probe_dir,
            probe_local,
            testing=testing,
            label=label,
            skip_remove_opto=skip_remove_opto,
        )

        # ======= Remove temporary SI folder ========= #
        # if rm_intermediate:
        #     try:
        #         shutil.rmtree(probe_local.joinpath(".si"))
        #     except Exception:
        #         _log.error("Could not delete temp si folder")

        # ======= Move to destination ========= #
        if move_final:
            phy_dest = probe_dest.joinpath(phy_local.name)

            if phy_dest.exists():
                _log.warning(f"Not moving because target {phy_dest} already exists")
            else:
                _log.info(f"Moving sorted data from {phy_local} to {phy_dest}")
                shutil.move(str(phy_local), str(phy_dest))

    # ======= Remove temporary SI folder ========= #
    if move_final and rm_intermediate and n_probes > 0:
        _log.info(f"Removing {session_local}")
        try:
            shutil.rmtree(session_local)
        except FileNotFoundError as e:
            _log.error(f"Could not delete {session_local}")
            _log.error(e)

        shutil.rmtree(SCRATCH_DIR)


if __name__ == "__main__":
    cli()
