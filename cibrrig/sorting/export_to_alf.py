import numpy as np
import shutil
import pandas as pd
import sys
from pathlib import Path
from spikeinterface.curation.model_based_curation import auto_label_units
from spikeinterface.core import write_binary_recording
from spikeinterface.exporters import export_to_ibl_gui
import warnings
from spikeinterface.core import SortingAnalyzer, BaseRecording
import spikeinterface.curation as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import logging

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

# QC presets
AMPLITUDE_CUTOFF = 0.1
SLIDING_RP = 0.1
AMP_THRESH = 40.0
MIN_SPIKES = 500

HUGGING_FACE_LOCAL = Path(r"C:\helpers\hugging_face")


def test_unit_refine_model_import():
    """
    Test if the UnitRefine model is available.
    """
    model_available = False
    try:
        model, model_info = sc.load_model(
            repo_id="SpikeInterface/UnitRefine_noise_neural_classifier",
            trusted=["numpy.dtype"],
        )
        model_available = True
        _log.info("UnitRefine model loaded from Hugging Face.")
    except Exception as e:
        _log.info(
            f"UnitRefine model not available from Hugging Face. Error: {e}. \nTrying local path {HUGGING_FACE_LOCAL.joinpath('UnitRefine_noise')}"
        )
        try:
            model, model_info = sc.load_model(
                model_folder=HUGGING_FACE_LOCAL.joinpath("UnitRefine_noise"),
                trust_model=True,
            )
            model_available = True
            _log.info("UnitRefine model loaded from local path.")
        except Exception as e:
            _log.error(
                "UnitRefine model not available. Please check your internet connection or local model path."
            )
            _log.error(e)

    if not model_available:
        _log.error(
            "UnitRefine model not available. Will skip UnitRefine autolabelling (NOT RECOMMENDED)."
        )


class ALFExporter:
    def __init__(
        self,
        analyzer: SortingAnalyzer,
        dest: Path,
        bin_path: Path | None = None,
        copy_binary: bool = False,
        job_kwargs: dict = dict(n_jobs=1, chunk_size="1s"),
        testing: bool = False,
    ):
        """
        Initialize the ALFExporter.
        Saves all the data in ALF format while still allowing for phy curation

        Applies the UnitRefine model for automated curation.
        Also uses IBL QC metrics to create bitwise_fail and label metrics


        Args:
            analyzer (SpikeInterfaceAnalyzer): The analyzer object.
            dest (Path): The path where the sorted data in ALF format are to be saved.
            job_kwargs (dict, optional): Job parameters for parallel processing.
        """
        self.analyzer = analyzer
        self.alf_path = dest
        self.bin_path = bin_path
        self.copy_binary = copy_binary
        self.job_kwargs = job_kwargs
        self.testing = testing
        self.templates = self.analyzer.get_extension("templates")
        self.used_sparsity = self.templates.sparsity
        self.sparse_templates = self.used_sparsity.sparsify_templates(
            self.templates.get_data()
        )
        self.get_channel_indices()
        self.get_lfp_recording()

    def get_channel_indices(self):
        """
        Get the channel indices for each unit.
        """
        n_chans = self.used_sparsity.max_num_active_channels
        used_channels = self.analyzer.channel_ids_to_indices(self.analyzer.channel_ids)
        channel_list = []
        
        for x in self.used_sparsity.unit_id_to_channel_indices.values():
            n_chans_missing = n_chans - len(x)
            if n_chans_missing > 0:
                min_chan = min(x)
                max_chan = max(x)
                if (min_chan-n_chans_missing) >=0:
                    append_chans = np.arange(min_chan-n_chans_missing, min_chan)
                else:
                    append_chans = np.arange(max_chan+1, max_chan+1+n_chans_missing)
                x = np.concatenate((x, append_chans))
            assert len(x) == n_chans, f"Channel indices length mismatch for unit {x}"
            channel_list.append(x)
        channel_indices = np.vstack(channel_list)

        assert np.all(np.isin(np.unique(channel_indices), used_channels)), "Channel indices contain invalid channels."
        assert channel_indices.shape[1] == n_chans, "Channel indices shape mismatch."
            
        self.channel_indices = channel_indices

    def get_lfp_recording(self):
        """
        Get the LFP recording if available.
        """
        if self.bin_path is None:
            self.lfp_recording = None
            return
        try:
            stream = se.get_neo_streams("spikeglx", self.bin_path)[0][0].replace(
                "lf", "ap"
            )
            lfp_recording = se.read_spikeglx(self.bin_path, stream_id=stream)
            if self.testing:
                lfp_recording = lfp_recording.frame_slice(
                    0, lfp_recording.get_sampling_frequency() * 60
                )
            lfp_recording = spre.bandpass_filter(
                lfp_recording, freq_min=1, freq_max=300
            )
            lfp_recording = spre.decimate(lfp_recording, 10)
            self.lfp_recording = lfp_recording
        except Exception as e:
            _log.error("Error loading LFP recording. Proceeding without LFP.")
            _log.error(e)
            self.lfp_recording = None

    def save_templates(self):
        """
        Save templates in ALF format

        Cluster waveforms and template waveforms are identical
        """
        np.save(self.alf_path.joinpath("clusters.waveforms.npy"), self.sparse_templates)
        np.save(
            self.alf_path.joinpath("clusters.waveformsChannels.npy"),
            self.channel_indices,
        )

        np.save(
            self.alf_path.joinpath("templates.waveforms.npy"), self.sparse_templates
        )
        np.save(
            self.alf_path.joinpath("templates.waveformsChannels.npy"),
            self.channel_indices,
        )
        shutil.copy(
            self.alf_path.joinpath("spikes.clusters.npy"),
            self.alf_path.joinpath("spikes.templates.npy"),
        )

        spike_samples = self.analyzer.sorting.to_spike_vector()["sample_index"]
        np.save(self.alf_path.joinpath("spikes.samples.npy"), spike_samples)

    def _apply_unit_refine_labels(self):
        """
        Apply UnitRefine (https://spikeinterface.readthedocs.io/en/stable/tutorials/curation/plot_1_automated_curation.html#sphx-glr-tutorials-curation-plot-1-automated-curation-py)
        to auto label the noise, multi-unit activity (MUA), and single-unit activity (SUA) clusters.
        """
        if sys.platform == "linux":
            noise_neuron_labels = auto_label_units(
                sorting_analyzer=self.analyzer,
                repo_id="SpikeInterface/UnitRefine_noise_neural_classifier",
                trusted=["numpy.dtype"],
            )
            noise_units = noise_neuron_labels[
                noise_neuron_labels["prediction"] == "noise"
            ]
            analyzer_neural = self.analyzer.remove_units(noise_units.index)

            sua_mua_labels = auto_label_units(
                sorting_analyzer=analyzer_neural,
                repo_id="SpikeInterface/UnitRefine_sua_mua_classifier",
                trusted=["numpy.dtype"],
            )
        else:
            noise_neuron_labels = auto_label_units(
                sorting_analyzer=self.analyzer,
                model_folder=Path(r"C:\helpers\hugging_face\UnitRefine_noise"),
                trust_model=True,
            )
            noise_units = noise_neuron_labels[
                noise_neuron_labels["prediction"] == "noise"
            ]
            analyzer_neural = self.analyzer.remove_units(noise_units.index)

            sua_mua_labels = auto_label_units(
                sorting_analyzer=analyzer_neural,
                model_folder=Path(r"C:\helpers\hugging_face\UnitRefine_sua"),
                trust_model=True,
            )

        unitrefine_label = (
            pd.concat([sua_mua_labels, noise_units]).sort_index().reset_index(drop=True)
        )
        unitrefine_label.rename(
            columns={"prediction": "UR_prediction", "probability": "UR_probability"},
            inplace=True,
        )
        unitrefine_label.index.name = "cluster_id"
        return unitrefine_label

    def create_full_metrics(self):
        """
        Extract the metrics from the analyzer, join them together, apply the unitrefine model, and save
        """
        # Move over metrics
        metrics = self.analyzer.get_extension("quality_metrics").get_data()
        template_metrics = self.analyzer.get_extension("template_metrics").get_data()
        metrics = metrics.merge(
            template_metrics, left_index=True, right_index=True, how="left"
        )
        try:
            ur_labels = self._apply_unit_refine_labels()
            ur_labels.index = metrics.index
            metrics = pd.concat([metrics, ur_labels], axis=1)
        except Exception as e:
            _log.error("Error computing model-based labels. Skipping.")
            _log.error(e)

        metrics.index.name = "si_unit_id"
        metrics = metrics.reset_index()
        metrics.index.name = "cluster_id"
        bitwise_pass = metrics.eval(
            "amplitude_cutoff<@AMPLITUDE_CUTOFF & sliding_rp_violation<@SLIDING_RP & abs(amplitude_median) > @AMP_THRESH & num_spikes>@MIN_SPIKES "
        )
        bitwise_pass = bitwise_pass.fillna(False)
        metrics["bitwise_fail"] = np.logical_not(bitwise_pass)
        metrics["label"] = bitwise_pass.astype(int)
        metrics.to_csv(self.alf_path.joinpath("clusters.metrics.csv"))

    # Compute and save PCs
    # TODO: Test (ripped from spikeinterface)
    def create_pca_features(self):
        """
        Extract the PC features from the analzyer and save to the target folder.

        This takes a while.
        """
        templates = self.analyzer.get_extension("templates")
        used_sparsity = templates.sparsity
        pca_extension = self.analyzer.get_extension("principal_components")
        pca_extension.run_for_all_spikes(
            self.alf_path.joinpath("pc_features.npy"), **self.job_kwargs
        )
        max_num_channels_pc = max(
            len(chan_inds)
            for chan_inds in used_sparsity.unit_id_to_channel_indices.values()
        )
        non_empty_units = []
        empty_flag = False
        for unit in self.analyzer.sorting.unit_ids:
            if len(self.analyzer.sorting.get_unit_spike_train(unit)) > 0:
                non_empty_units.append(unit)
            else:
                empty_flag = True

        if empty_flag:
            warnings.warn("Empty units have been removed while exporting to ALF")
        unit_ids = non_empty_units

        if len(unit_ids) == 0:
            raise Exception(
                "No non-empty units in the sorting result, can't save to ALF."
            )

        unit_ids = non_empty_units
        pc_feature_ind = -np.ones((len(unit_ids), max_num_channels_pc), dtype="int64")
        for unit_ind, unit_id in enumerate(unit_ids):
            chan_inds = used_sparsity.unit_id_to_channel_indices[unit_id]
            pc_feature_ind[unit_ind, : len(chan_inds)] = chan_inds
        np.save(self.alf_path.joinpath("pc_feature_ind.npy"), pc_feature_ind)

    def save_extracted_waveforms(self):
        """
        Write already extracted waveforms to target folder
        """
        wvfms = self.analyzer.get_extension("waveforms")
        sv = self.analyzer.get_extension("random_spikes").get_random_spikes()
        spike_indices = sv["sample_index"]
        spike_clusters = sv["unit_index"]
        chans_subset = np.zeros(
            (len(spike_indices), self.channel_indices.shape[1]), dtype="int16"
        )
        for clu in np.unique(spike_clusters):
            idx = np.where(spike_clusters == clu)[0]
            chans_subset[idx] = self.channel_indices[clu]
        np.save(
            self.alf_path.joinpath("_phy_spikes_subset.waveforms.npy"),
            wvfms.get_data(),
        )
        np.save(
            self.alf_path.joinpath("_phy_spikes_subset.spikes.npy"),
            self.analyzer.get_extension("random_spikes").get_data(),
        )
        np.save(self.alf_path.joinpath("_phy_spikes_subset.channels.npy"), chans_subset)

    def write_params(self):
        """
        Copy the binary and write the params.py file.
        The params.py file will point to the local recording.
        """
        num_chans = self.analyzer.recording.get_num_channels()
        dtype = self.analyzer.get_dtype()
        dtype_str = np.dtype(dtype).name
        fs = self.analyzer.recording.get_sampling_frequency()
        rec_path = self.alf_path.joinpath("recording.dat")

        # Do this dance with the n_jobs because it is faster on the HPC to save with one job.
        if self.copy_binary:
            if sys.platform == "linux":
                n_jobs_stash = self.job_kwargs["n_jobs"]
                self.job_kwargs["n_jobs"] = 1
            write_binary_recording(
                self.analyzer.recording,
                file_paths=rec_path,
                dtype=dtype,
                **self.job_kwargs,
            )
            if sys.platform == "linux":
                self.job_kwargs["n_jobs"] = n_jobs_stash

        with (self.alf_path / "params.py").open("w") as f:
            f.write("dat_path = r'recording.dat'\n")
            f.write(f"n_channels_dat = {num_chans}\n")
            f.write(f"dtype = '{dtype_str}'\n")
            f.write("offset = 0\n")
            f.write(f"sample_rate = {fs}\n")
            f.write(f"hp_filtered = {self.analyzer.is_filtered()}")

    def run(self):
        """
        Run all steps to export sorting as ALF/IBL structure
        """
        if self.alf_path.exists():
            _log.critical(
                f" Target sorted folder {self.alf_path} already exists. Please remove it before exporting."
            )
            return
        export_to_ibl_gui(
            sorting_analyzer=self.analyzer,
            output_folder=self.alf_path,
            lfp_recording=self.lfp_recording,
        )
        self.save_templates()
        self.create_full_metrics()
        self.write_params()
        self.create_pca_features()
        self.save_extracted_waveforms()
