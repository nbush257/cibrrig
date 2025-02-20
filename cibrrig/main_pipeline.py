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
from pathlib import Path
import shutil
import sys
import one.alf.io as alfio
from cibrrig.utils.alf_utils import Recording
import json
from cibrrig.archiving import backup, ephys_data_to_alf
from cibrrig.preprocess import preproc_pipeline
from cibrrig.sorting import spikeinterface_ks4
from cibrrig.gui import DirectorySelector, OptoFileFinder, WiringEditor,OptoInsertionTableApp, NpxInsertionTableApp,NotesDialog,plot_probe_insertion, plot_insertion_layout
import subprocess
import pandas as pd


#TODO: Solve depth issue with insertions

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
        num_opto_fibers
    ) = window.get_paths()

    n_gates = len(gate_paths)
    notes_fn = local_run_path.joinpath("notes.json")
    notes = NotesDialog(n_gates,notes_fn)
    notes.save_notes()


    # Get the opto calibrations and wiring files
    for gate in gate_paths:
        opto_fn = list(gate.glob("opto_calibration.json"))
        if not opto_fn:
            opto_finder = OptoFileFinder(title=f'{gate.stem}')
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
    insertions = pd.DataFrame()
    for ii in range(num_probes):
        name = f'imec{ii}'
        insertion_table = NpxInsertionTableApp(name=name)
        insertion_table.exec_()
        _insertions = insertion_table.get_insertions()
        _insertions['probe'] = f'imec{ii}'
        _insertions.to_csv(local_run_path.joinpath(f'_cibrrig_{name}.insertionsManipulator.csv'))

        insertions = pd.concat([insertions, _insertions])

    for ii in range(num_opto_fibers):
        name = f'opto{ii}'
        insertion_table = OptoInsertionTableApp(name=name)
        insertion_table.exec_()
        _insertions = insertion_table.get_insertions()
        _insertions['probe'] = f'opto{ii}'
        _insertions.to_csv(local_run_path.joinpath(f'_cibrrig_{name}.insertionsManipulator.csv'))

        insertions = pd.concat([insertions, _insertions])

    save_fn = Path(local_run_path).joinpath("caudal_insertion_map.png")
    plot_insertion_layout(insertions,save_fn)
    save_fn = Path(local_run_path).joinpath("all_insertions.png")
    plot_probe_insertion(insertions,save_fn)


    # RUN BACKUP
    backup.no_gui(local_run_path, remote_archive_path)

    # RUN RENAME
    ephys_data_to_alf.run(local_run_path)

    # Get all sessions
    sessions_paths = list(alfio.iter_sessions(local_run_path))
    sessions_paths.sort()
    for session in sessions_paths:
        # RUN EXTRACT AND PREPROCESS
        skip_ephysQC = not run_ephysQC
        preproc_pipeline.run(session, skip_ephysQC)
        rec = Recording(session)
        rec.concatenate_session()

        # RUN SPIKESORTING
        skip_remove_opto = not remove_opto_artifact
        spikeinterface_ks4.run(session, skip_remove_opto=skip_remove_opto)
        params_files = session.rglob("params.py")

        # PHY EXTRACT WAVEFORMS
        for pp in params_files:
            command = ["phy", "extract-waveforms", pp]
            subprocess.run(command, cwd=pp.parent)
    # Move all data to RSS
    shutil.move(local_run_path, remote_working_path)


if __name__ == "__main__":
    main()
