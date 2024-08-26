import one.alf.io as alfio
from pathlib import Path
import numpy as np
import spikeglx
import logging
import pandas as pd
from ..preprocess.extract_opto_times import load_opto_calibration

logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


# These are all the potential keys that contain time relative to experiment start
# and thus need to be modified during concatenation
TEMPORAL_FEATURES = [
    "intervals",
    "times",
    "on_sec",
    "off_sec",
    "pk_time",
    "inhale_peaks",
    "exhale_troughs",
    "inhale_onsets",
    "exhale_onsets",
    "inhale_offsets",
    "exhale_offsets",
    "inhale_pause_onsetes",
    "exhale_pause_onsets",
    "start_time",
    "end_time",
]


class Recording:
    def __init__(self, session_path):
        self.session_path = Path(session_path)
        self.raw_ephys_path = self.session_path.joinpath("raw_ephys_data")
        assert (
            self.raw_ephys_path.is_dir()
        ), "Raw ephys data folder does not exist as expected"
        self.alf_path = self.session_path.joinpath("alf")
        assert self.alf_path.is_dir(), "Alf path does not exist as expected"
        self.ni_fns = list(self.raw_ephys_path.glob("*nidq.bin"))
        self.ni_fns.sort()
        self.n_recs = len(self.ni_fns)
        _log.info(
            f"Found {self.n_recs} recordings:\n\t"
            + "\n\t".join([x.name for x in self.ni_fns])
        )
        self.get_breaks_times()
        self.list_all_alf_objects()

    def get_breaks_times(self):
        # I think I am repeating myself here
        breaks = [0]
        for ni_fn in self.ni_fns:
            SR = spikeglx.Reader(ni_fn)
            breaks.append(breaks[-1] + SR.meta["fileTimeSecs"])
        self.breaks = np.array(breaks)

    def list_all_alf_objects(self):
        object_parts = alfio.filter_by(self.alf_path)[1]
        object_names = list(set([x[1] for x in object_parts]))
        object_names.sort()
        self.alf_objects = object_names

    def concatenate_triggers(self, object_name):
        try:
            alf_obj_out = alfio.load_object(
                self.alf_path, object_name, extra="t0", short_keys=True
            )
        except Exception:
            # Doing it this way so that if the files can be loaded, we do not error, but do throw an error if the problem is not the trigger label
            alf_obj_out = alfio.load_object(self.alf_path, object_name, short_keys=True)
            _log.info("Triggers not found. Has this already been concatenated?")
            return alf_obj_out

        for ii in range(1, self.n_recs):
            try:
                alf_obj = alfio.load_object(
                    self.alf_path, object_name, extra=f"t{ii:0.0f}", short_keys=True
                )
            except Exception:
                _log.warning(
                    f"{object_name} for trigger {ii} not found. If this was recorded then this is an issue."
                )
                continue
            for k in alf_obj.keys():
                if len(alf_obj[k]) == 0:
                    continue
                if k in TEMPORAL_FEATURES:
                    alf_obj[k] += self.breaks[ii]
                alf_obj_out[k] = np.concatenate([alf_obj_out[k], alf_obj[k]])
        return alf_obj_out

    def concatenate_log(self, save=True, overwrite=True):
        try:
            alf_obj_out = alfio.load_object(
                self.session_path, "log", extra="t0", short_keys=True
            )
            log_df_out = alf_obj_out.to_df()
        except Exception:
            # Doing it this way so that if the files can be loaded, we do not error, but do throw an error if the problem is not the trigger label
            alf_obj_out = alfio.load_object(self.session_path, "log", short_keys=True)
            _log.info("Triggers not found. Has this already been concatenated? Or is this a one-off run?")
            log_df_out = alf_obj_out.to_df()
            _log.info(f'{log_df_out}')
            # return(alf_obj_out)

        old_files, old_file_parts = alfio.filter_by(self.session_path, object="log")
        for ii in range(1, self.n_recs):
            alf_obj = alfio.load_object(
                self.session_path, "log", extra=f"t{ii:0.0f}", short_keys=True
            )
            log_df = alf_obj.to_df()
            for k in log_df.keys():
                if len(log_df[k]) == 0:
                    continue
                if k in TEMPORAL_FEATURES:
                    log_df[k] += self.breaks[ii]

            log_df_out = pd.concat([log_df_out, log_df])
        log_df_out.reset_index(inplace=True, drop=True)
        log_df_out = self.improve_opto_log(log_df_out)
        if save:
            table_fn = alfio.files.spec.to_alf(
                "log", "table", extension="tsv", namespace="cibrrig"
            )
            log_df_out.drop("Unnamed: 0", axis=1, inplace=True)
            log_df_out.to_csv(
                self.session_path.joinpath(table_fn), sep="\t", index=None
            )

        if overwrite:
            _log.info("Removing old files")
            _log.debug("\n\t" + "\n\t".join(old_files))
            for fn in old_files:
                self.session_path.joinpath(fn).unlink()

        return alf_obj_out

    def improve_opto_log(self, log):
        """
        Grab the amplitudes from the opto json and add scale to log
        Make pulses have an end time of the pulse duration
        """

        opto_calib = load_opto_calibration(self.session_path)
        amps_mw = opto_calib(log["amplitude"].values.astype("f"))
        log["amplitude_mw"] = amps_mw
        log.loc[log["label"] == "opto_pulse", "end_time"] = (
            log["start_time"] + log["duration"]
        )

        return log

    def concatenate_alf_objects(self, save=True, overwrite=True):
        for object_name in self.alf_objects:
            if object_name == "log":
                _log.debug(
                    "Do not concatenate log here - it requires special handling"
                )
                continue
            _log.info(f"Concatenating {object_name}")
            obj_cat = self.concatenate_triggers(object_name)
            old_files, old_file_parts = alfio.filter_by(
                self.alf_path, object=object_name
            )
            if old_file_parts[0][-2] is None:
                _log.info(
                    f"Skipping concatenation of {object_name}. It appears to be concatenated"
                )
                continue

            extensions = list(set([x[-1] for x in old_file_parts]))
            namespaces = list(set([x[0] for x in old_file_parts]))
            attributes = list(set([x[2] for x in old_file_parts]))
            assert (
                len(namespaces) == 1
            ), f"Multiple namespaces found for {object_name}:{namespaces}"
            namespace = namespaces[0]

            if save:
                if "table" in attributes:
                    table_fn = alfio.files.spec.to_alf(
                        object_name, "table", extension="pqt", namespace=namespace
                    )
                    _log.info(f"Saving concatenated {object_name} to parquet")
                    obj_cat.to_df().to_parquet(self.alf_path.joinpath(table_fn))
                else:
                    assert len(extensions) == 1, "There should only be one extension"
                    assert extensions[0] == "npy", "There should only be one extension"
                    _log.info(f"Saving concatenated {object_name} to npy")
                    alfio.save_object_npy(
                        self.alf_path, obj_cat, object_name, namespace=namespace
                    )
            if overwrite:
                _log.info("Removing old files")
                _log.debug("\n\t" + "\n\t".join(old_files))
                for fn in old_files:
                    self.alf_path.joinpath(fn).unlink()

    def concatenate_session(self, save=True, overwrite=True):
        self.concatenate_alf_objects(save=save, overwrite=overwrite)
        try:
            self.concatenate_log(save=save, overwrite=overwrite)
        except Exception as e:
            _log.error(e)
            _log.warning(
                "Log file concatenation failure. This should not happen on new data."
            )

    def load_spikes():
        # TODO: Only keep good cells?
        # TODO: Work with multiple probes?
        pass
