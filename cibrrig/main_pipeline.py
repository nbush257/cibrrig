"""
Takes a spikeglx run and performs:
1) compression and copy backup to archive
2) Renaming to alf
3) Preprocessing
4) Spikesorting

# Gui created in part by chatgpt
"""

from PyQt5.QtWidgets import (
    QApplication,
)
import os

# os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import shutil
import sys
import one.alf.io as alfio
from cibrrig.utils.alf_utils import Recording
import json
from cibrrig.archiving import backup, ephys_data_to_alf
from cibrrig.preprocess import preproc_pipeline
from cibrrig.sorting import spikeinterface_ks4
from cibrrig.gui import (
    DirectorySelector,
    OptoFileFinder,
    WiringEditor,
    OptoInsertionTableApp,
    NpxInsertionTableApp,
    NotesDialog,
    plot_probe_insertion,
    plot_insertion_layout,
)
import pandas as pd
import logging
import re
import enum
from cibrrig.sorting.export_to_alf import test_unit_refine_model_import
from PyQt5.QtWidgets import QMessageBox
import cibrrig.postprocess.synchronize_sorting_to_aux as sync_aux
import cibrrig.utils.utils as utils
import click
from cibrrig.postprocess import extract_resp_modulation


class Status(enum.IntEnum):
    NONE = 0
    PREPROC = 10
    SPIKESORTED = 20
    CONCATENATED = 30
    SYNCHRONIZED = 40
    RESP_MOD_COMPUTED = 50


# TODO: Solve depth issue with insertions


def check_is_alf(run_path, gate_paths):
    """Check if the run path is already in ALF format"""
    sub_dirs = [x for x in run_path.iterdir() if x.is_dir()]
    # check to see if the subdirs are of the form "YYYY-MM-DD" using regex
    sub_dirs = [x for x in sub_dirs if re.match(r"\d{4}-\d{2}-\d{2}", x.name)]

    if len(sub_dirs) > 0 and len(gate_paths) == 0:
        is_alf = True
    elif len(sub_dirs) == 0 and len(gate_paths) > 0:
        is_alf = False
    elif len(sub_dirs) > 0 and len(gate_paths) > 0:
        logging.warning("Both ALF and non-ALF directories found!")
    elif len(sub_dirs) == 0 and len(gate_paths) == 0:
        logging.error("No data found")
    else:
        raise ValueError("Something funny about the run directory")

    return is_alf


def set_status(session, status):
    """Set the status of the session"""
    status_fn = session.joinpath("status.json")
    with open(status_fn, "w") as fid:
        json.dump({"status": status}, fid)


def get_status(session):
    """Get the status of the session"""
    status_fn = session.joinpath("status.json")
    if status_fn.exists():
        with open(status_fn, "r") as fid:
            status = json.load(fid)
            status = status["status"]
    else:
        status = Status.NONE

    print(f"Status of {session}: {status}")
    return status


def check_unit_refine():
    has_unit_refine = test_unit_refine_model_import()
    if not has_unit_refine:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Unit Refine Model Not Found")
        msg.setText(
            "The unit refine model could not be imported. Model based unit labelling will be skipped."
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


def setup_logging(local_run_path):
    log_file = local_run_path.joinpath("cibrrig_pipeline.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s : %(message)s"
)
    file_handler.setFormatter(formatter)
    _log = logging.getLogger()
    _log.setLevel(logging.INFO)
    # Remove all handlers associated with the root logger object (avoid duplicate logs)
    for handler in _log.handlers[:]:
        _log.removeHandler(handler)
    _log.addHandler(file_handler)
    # Add a handler to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    _log.addHandler(console_handler)
    return _log


def main():
    """Main function to run the pipeline

    This function runs the pipeline for a selected run directory. It performs the following steps:
    1) Backup and compress the data to the archive
    2) Rename the data to ALF format
    3) Extract and preprocess the auxiliary data
    4) Spikesort the data
    5) Move the data to the working directory
    """
    app = QApplication(sys.argv)
    window = DirectorySelector()
    window.show()
    app.exec_()

    # After the GUI is closed, retrieve the selected paths
    (
        local_run_path,
        remote_archive_path,
        remote_working_path,
        remove_opto_artifact,
        run_ephysQC,
        gate_paths,
        num_probes,
        num_opto_fibers,
    ) = window.get_paths()

    check_unit_refine()

    log_fn = local_run_path.joinpath("cibrrig.log")
    logging.basicConfig(filename=log_fn, level=logging.INFO)
    is_alf = check_is_alf(local_run_path, gate_paths)
    n_gates = len(gate_paths)
    if not is_alf:
        notes_fn = local_run_path.joinpath("notes.json")
        notes = NotesDialog(n_gates, notes_fn)
        notes.exec_()
        notes.save_notes()

        for gate in gate_paths:
            opto_fn = list(gate.glob("opto_calibration.json"))
            if not opto_fn:
                opto_finder = OptoFileFinder(title=f"{gate.stem}")
                opto_finder.exec_()
                opto_fn = opto_finder.get_opto_file()
                print(opto_fn)
                if opto_fn.name != "":
                    print(f"Copying {opto_fn}")
                    shutil.copy(opto_fn, gate)
                else:
                    print("Skipping opto file")

            # Get the wiring.json files
            wiring_fn = list(gate.glob("*nidq.wiring.json"))
            if not wiring_fn:
                wiring_fn = gate.joinpath("nidq.wiring.json")
                wiring_editor = WiringEditor(title=gate.stem)
                wiring_editor.exec_()
                wiring = wiring_editor.get_output_wiring()
                with open(wiring_fn, "w") as fid:
                    json.dump(wiring, fid)
                print("Created wiring file")

        # Get insertions and save
        for ii in range(num_probes):
            name = f"imec{ii}"
            fn = local_run_path.joinpath(f"_cibrrig_{name}.insertionsManipulator.csv")
            insertion_table = NpxInsertionTableApp(
                name=name, n_gates=n_gates, save_fn=fn
            )
            insertion_table.exec_()
            insertion_table.to_csv()

        for ii in range(num_opto_fibers):
            name = f"opto{ii}"
            fn = local_run_path.joinpath(f"_cibrrig_{name}.insertionsManipulator.csv")
            insertion_table = OptoInsertionTableApp(
                name=name, n_gates=n_gates, save_fn=fn
            )
            insertion_table.exec_()
            insertion_table.to_csv()

        # Load all insertions
        insertions = pd.DataFrame()
        insertions_fns = list(local_run_path.rglob("*.insertionsManipulator.csv"))
        for fn in insertions_fns:
            insertions = pd.concat([insertions, pd.read_csv(fn)])

        save_fn = Path(local_run_path).joinpath("caudal_insertion_map.png")
        plot_insertion_layout(insertions, save_fn)
        save_fn = Path(local_run_path).joinpath("all_insertions.png")
        plot_probe_insertion(insertions, save_fn)

        # RUN BACKUP

    # RUN MAIN PIPELINE
    run(
        local_run_path,
        remote_working_path,
        remote_archive_path,
        remove_opto_artifact,
        run_ephysQC,
        compress_locally=True,  # Use new compressed archive workflow by default
    )


def run(
    local_run_path: Path,
    remote_working_path: Path,
    remote_archive_path: Path,
    remove_opto_artifact: bool,
    run_ephysQC: bool,
    compress_locally: bool = True,
):
    """Run the main pipeline
    1) Compress data locally (if compress_locally=True)
    2) Backup compressed data to archive
    3) Rename to ALF if not already
    4) Preprocess each session
    5) Spikesort each session
    6) Move to working directory
    7) Synchronize sorting to aux

    Args:
        local_run_path (Path): Path to the local run directory
        remote_working_path (Path): Path to the remote working directory
        remote_archive_path (Path): Path to the remote archive directory
        remove_opto_artifact (bool): Whether to remove opto artifact during preprocessing
        run_ephysQC (bool): Whether to run ephys QC during preprocessing
        compress_locally (bool): Whether to compress data locally before backup. Defaults to True.
    """
    # Configure logging to a file handler for the session
    _log = setup_logging(local_run_path)
    _log.info(f"{local_run_path = }")
    _log.info(f"{remote_working_path = }")
    _log.info(f"{remote_archive_path = }")
    _log.info(f"{remove_opto_artifact = }")
    _log.info(f"{run_ephysQC = }")
    _log.info("Starting pipeline")

    is_gate, local_run_path = utils.check_is_gate(local_run_path, move_if_gate=True)

    gate_paths = utils.get_gates(local_run_path)
    is_alf = check_is_alf(local_run_path, gate_paths)
    if not is_alf:
        _log.debug("Not ALF Format")
        _log.info(f"Backing up local data to {remote_archive_path}")
        # Pass compress_locally parameter to backup function
        backup.no_gui(local_run_path, remote_archive_path, compress_locally=compress_locally)
        # RUN RENAME
        _log.info("Renaming to ALF format")
        ephys_data_to_alf.run(local_run_path)

    # Get all sessions
    sessions_paths = list(alfio.iter_sessions(local_run_path))
    sessions_paths.sort()
    skip_ephysQC = not run_ephysQC
    skip_remove_opto = not remove_opto_artifact
    _log.debug(f"Found {len(sessions_paths)} sessions: {sessions_paths}")
    for session in sessions_paths:
        # RUN EXTRACT AND PREPROCESS
        status = get_status(session)
        _log.debug(f"Session {session}: {status = }")
        if status < Status.PREPROC:
            _log.info(f"Running preprocessing for {session}")
            preproc_pipeline.run(session, skip_ephysQC)
            set_status(session, Status.PREPROC)

        # RUN SPIKESORTING
        if status < Status.SPIKESORTED:
            _log.info(f"Running spikesorting for {session}")
            spikeinterface_ks4.run(session, skip_remove_opto=skip_remove_opto)
            set_status(session, Status.SPIKESORTED)

        # RUN CONCATENATION
        if status < Status.CONCATENATED:
            _log.info(f"Running concatenation for {session}")
            rec = Recording(session)
            rec.concatenate_session()
            set_status(session, Status.CONCATENATED)

        # RUN SYNCHRONIZATION TO AUX
        if status < Status.SYNCHRONIZED:
            _log.info(f"Running synchronization to aux for {session}")
            sync_aux.run_session(session)
            set_status(session, Status.SYNCHRONIZED)

        # RUN EXTRACT RESP MODULATION
        if status < Status.RESP_MOD_COMPUTED:
            _log.info(f"Running respiratory modulation extraction for {session}")
            rez = extract_resp_modulation.run_session(session)
            if rez == 0:
                _log.info(f"Respiratory modulation computed for {session}")
                set_status(session, Status.RESP_MOD_COMPUTED)
            else:
                _log.warning(f"Respiratory modulation not computed for {session}")

    # Move all data to RSS
    _log.info(f"Moving data to working directory {remote_working_path}")
    shutil.move(local_run_path, remote_working_path)


@click.command()
@click.argument("local_run_path", type=click.Path(exists=True))
@click.argument("remote_working_path", type=click.Path())
@click.argument("remote_archive_path", type=click.Path())
@click.option(
    "--remove_opto_artifact",
    "-O",
    is_flag=True,
    help="Remove opto artifact during preprocessing",
)
@click.option(
    "--run_ephysqc", "-Q", is_flag=True, help="Run ephys QC during preprocessing"
)
@click.option("--no_local_compression", is_flag=True, help= 'Use legacy remote compression instead of local compression')
def cli(
    local_run_path,
    remote_working_path,
    remote_archive_path,
    remove_opto_artifact=False,
    run_ephysqc=False,
    no_local_compression=False):
    """
    Command line interface for running the main pipeline.

    Args:
        local_run_path (str): Path to the data source run directory. Typically on the local computer (NPX acquisition)
        remote_working_path (str): Path to the remote working directory where uncompressed active data is stored
        remote_archive_path (str): Path to the remote archive directory where compressed freezes are stored
        remove_opto_artifact (bool): Whether to remove opto artifact during preprocessing
        run_ephysQC (bool): Whether to run ephys QC during preprocessing
        no_local_compression (bool): Whether to use legacy remote compression behavior

    Returns:
        None
    """
    local_run_path = Path(local_run_path)
    remote_working_path = Path(remote_working_path)
    remote_archive_path = Path(remote_archive_path)
    run(
        local_run_path,
        remote_working_path,
        remote_archive_path,
        remove_opto_artifact,
        run_ephysqc,
        compress_locally=not no_local_compression,
    )


if __name__ == "__main__":
    cli()

#TODO: Preproc and spikesort from archived cbin