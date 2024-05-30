'''
Takes a spikeglx run and performs:
1) compression and copy backup to archive 
2) Renaming to alf
3) Preprocessing
4) Spikesorting

# Gui created by chatgpt
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QCheckBox, QGridLayout, QDialog,QGroupBox,QComboBox,QHBoxLayout,QInputDialog
from PyQt5.QtCore import Qt, pyqtSignal
from pathlib import Path
import shutil
import sys
import one.alf.io as alfio
import shutil
from .utils.alf_utils import Recording
import json
from .archiving import backup,ephys_data_to_alf
from .preprocess import preproc_pipeline
from .sorting import spikeinterface_ks4

DEFAULT_WIRING = {
    "SYSTEM": "3B",
    "SYNC_WIRING_DIGITAL": {
        "P0.0": "record",
        "P0.7": "imec_sync"
    },
    "SYNC_WIRING_ANALOG": {
        "AI0": "diaphragm",
        "AI1": "ekg",
        "AI4": "pdiff",
        "AI5": "laser",
        "AI6": "diaphragm_integrated",
        "AI7": "temperature"
    }
}

# List of Sync inputs that are expected
POSSIBLE_WIRINGS = {
    "Digital": ["None", "record", "imec_sync","right_camera", "Custom"], 
    "Analog": ["None", "diaphragm", "ekg", "pdiff", "laser", "diaphragm_integrated", "temperature", "Custom"]  
}

class DirectorySelector(QWidget):
    def __init__(self):
        super().__init__()
        self.local_run_path = Path("D:/sglx_data/Subjects")
        self.remote_archive_path = Path("U:/alf_data_repo/ramirez/Subjects")
        self.remote_working_path = Path("X:/alf_data_repo/ramirez/Subjects")
        self.remove_opto_artifact = True
        self.run_ephys_qc = True

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        grid_layout = QGridLayout()

        # Local run path selection
        self.local_run_label = QLabel("Local Run Path:")
        self.local_run_line_edit = QLineEdit(str(self.local_run_path))
        self.local_run_button = QPushButton("Browse...")
        self.local_run_button.clicked.connect(self.select_local_run_path)
        grid_layout.addWidget(self.local_run_label, 0, 0)
        grid_layout.addWidget(self.local_run_line_edit, 0, 1)
        grid_layout.addWidget(self.local_run_button, 0, 2)

        # Remote archive path selection
        self.remote_archive_label = QLabel("Remote Subjects Archive Path:")
        self.remote_archive_line_edit = QLineEdit(str(self.remote_archive_path))
        self.remote_archive_button = QPushButton("Browse...")
        self.remote_archive_button.clicked.connect(self.select_remote_archive_path)
        grid_layout.addWidget(self.remote_archive_label, 1, 0)
        grid_layout.addWidget(self.remote_archive_line_edit, 1, 1)
        grid_layout.addWidget(self.remote_archive_button, 1, 2)

        # Remote working path selection
        self.remote_working_label = QLabel("Remote Working Subjects Path:")
        self.remote_working_line_edit = QLineEdit(str(self.remote_working_path))
        self.remote_working_button = QPushButton("Browse...")
        self.remote_working_button.clicked.connect(self.select_remote_working_path)
        grid_layout.addWidget(self.remote_working_label, 2, 0)
        grid_layout.addWidget(self.remote_working_line_edit, 2, 1)
        grid_layout.addWidget(self.remote_working_button, 2, 2)

        # Checkbox for remove_opto_artifact
        self.remove_opto_artifact_checkbox = QCheckBox("Remove Opto Artifact")
        self.remove_opto_artifact_checkbox.setChecked(self.remove_opto_artifact)
        self.remove_opto_artifact_checkbox.setToolTip("Uncheck if stimulation is unlikely to cause light artifacts")
        self.remove_opto_artifact_checkbox.stateChanged.connect(self.toggle_remove_opto_artifact)
        grid_layout.addWidget(self.remove_opto_artifact_checkbox, 3, 0, 1, 3)

        # Checkbox for run_ephys_qc
        self.run_ephys_qc_checkbox = QCheckBox("Run EphysQC")
        self.run_ephys_qc_checkbox.setChecked(self.run_ephys_qc)
        grid_layout.addWidget(self.run_ephys_qc_checkbox, 4, 0, 1, 3)

        layout.addLayout(grid_layout)

        # Submit button
        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)
        self.setWindowTitle("Directory Selector")

    def select_local_run_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Local Run Path", str(self.local_run_path))
        if directory:
            self.local_run_path = Path(directory)
            self.local_run_line_edit.setText(str(self.local_run_path))

    def select_remote_archive_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Remote Subjects Archive Path", str(self.remote_archive_path))
        if directory:
            self.remote_archive_path = Path(directory)
            self.remote_archive_line_edit.setText(str(self.remote_archive_path))

    def select_remote_working_path(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Remote Subjects Working Path", str(self.remote_working_path))
        if directory:
            self.remote_working_path = Path(directory)
            self.remote_working_line_edit.setText(str(self.remote_working_path))

    def toggle_remove_opto_artifact(self, state):
        if state == Qt.Checked:
            self.remove_opto_artifact = True
        else:
            self.remove_opto_artifact = False

    def submit(self):
        self.close()  # Close the GUI window

    def get_paths(self):
        return self.local_run_path, self.remote_archive_path, self.remote_working_path, self.remove_opto_artifact, self.run_ephys_qc


class OptoFileFinder(QDialog):
    opto_file_selected = pyqtSignal(Path)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Selection")
        self.setGeometry(100, 100, 300, 100)

        layout = QVBoxLayout()

        label = QLabel("opto_calibration.json not found. Please select a file or skip.")
        layout.addWidget(label)

        self.select_button = QPushButton("Select File")
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        skip_button = QPushButton("Skip")
        skip_button.clicked.connect(self.skip_file)
        layout.addWidget(skip_button)

        self.setLayout(layout)

        self.opto_file = None  # Initialize opto_file attribute

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "JSON Files (*.json)")
        if file_path:
            self.opto_file = Path(file_path)
            self.opto_file_selected.emit(self.opto_file)  # Emit signal with opto file path
            self.close()

    def skip_file(self):
        self.opto_file = Path('')
        self.opto_file_selected.emit(self.opto_file)  # Emit signal with None (skipped)
        self.close()

    def get_opto_file(self):  # Method to get opto file after dialog is closed
        return self.opto_file


class WiringEditor(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dictionary Editor')
        self.setGeometry(100, 100, 600, 200)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Digital and Analog Groups
        groups_layout = QHBoxLayout()
        main_layout.addLayout(groups_layout)

        # Digital Group
        digital_group = QGroupBox('Digital')
        digital_layout = QVBoxLayout()

        self.digital_entries = {}
        digital_keys = [f'P0.{i}' for i in range(8)]
        for key in digital_keys:
            label = QLabel(key)
            combo_box = QComboBox()
            combo_box.addItems(POSSIBLE_WIRINGS.get("Digital", []))
            combo_box.setCurrentText(DEFAULT_WIRING.get("SYNC_WIRING_DIGITAL", {}).get(key, "None"))
            combo_box.currentIndexChanged.connect(lambda _, cb=combo_box, k=key: self.on_digital_value_changed(cb, k))
            digital_layout.addWidget(label)
            digital_layout.addWidget(combo_box)
            self.digital_entries[key] = combo_box

        digital_group.setLayout(digital_layout)
        groups_layout.addWidget(digital_group)

        # Analog Group
        analog_group = QGroupBox('Analog')
        analog_layout = QVBoxLayout()

        self.analog_entries = {}
        analog_keys = [f'AI{i}' for i in range(8)]
        for key in analog_keys:
            label = QLabel(key)
            combo_box = QComboBox()
            combo_box.addItems(POSSIBLE_WIRINGS.get("Analog", []))
            combo_box.setCurrentText(DEFAULT_WIRING.get("SYNC_WIRING_ANALOG", {}).get(key, "None"))
            combo_box.currentIndexChanged.connect(lambda _, cb=combo_box, k=key: self.on_analog_value_changed(cb, k))
            analog_layout.addWidget(label)
            analog_layout.addWidget(combo_box)
            self.analog_entries[key] = combo_box

        analog_group.setLayout(analog_layout)
        groups_layout.addWidget(analog_group)

        # Save Button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_values)
        main_layout.addWidget(save_button)

    def on_digital_value_changed(self, combo_box, key):
        if combo_box.currentText() == "Custom":
            text, ok = QInputDialog.getText(self, 'Custom Value', f'Enter custom value for {key}:')
            if ok:
                combo_box.addItem(text)
                combo_box.setCurrentText(text)

    def on_analog_value_changed(self, combo_box, key):
        if combo_box.currentText() == "Custom":
            text, ok = QInputDialog.getText(self, 'Custom Value', f'Enter custom value for {key}:')
            if ok:
                combo_box.addItem(text)
                combo_box.setCurrentText(text)

    def save_values(self):
        output_dictionary = {
            'SYSTEM':'3B',
            'SYNC_WIRING_ANALOG':{},
            'SYNC_WIRING_DIGITAL':{}
        }

        for key, combo_box in self.digital_entries.items():
            value = combo_box.currentText()
            if value != "None":
                output_dictionary['SYNC_WIRING_DIGITAL'][key] = value

        for key, combo_box in self.analog_entries.items():
            value = combo_box.currentText()
            if value != "None":
                output_dictionary['SYNC_WIRING_ANALOG'][key] = value

        print("Output Dictionary:", output_dictionary)
        self.output_wiring = output_dictionary
        self.close()
    def get_output_wiring(self):
        return(self.output_wiring)


def main():
    app = QApplication(sys.argv)
    window = DirectorySelector()
    window.show()
    app.exec_()

    # After the GUI is closed, retrieve the selected paths
    local_run_path, remote_archive_path, remote_working_path, remove_opto_artifact, run_ephysQC = window.get_paths()
    if local_run_path == 'D:/sglx_data/Subjects':
        raise ValueError('You picked the root Subjects folder. This is a scary thing to do and incorrect.')

    gate_paths = list(local_run_path.glob('*_g[0-9]*'))

    for gate in gate_paths:
        opto_fn = list(gate.glob('opto_calibration.json'))
        if not opto_fn:
            opto_finder = OptoFileFinder()
            opto_finder.exec_()
            opto_fn = opto_finder.get_opto_file()
            print(opto_fn)
            if opto_fn.name!='':
                print(f"Copying {opto_fn}")
                shutil.copy(opto_fn, gate)
            else:
                print('Skipping opto file')

    for gate in gate_paths:
        wiring_fn = list(gate.glob('nidq.wiring.json'))
        if not wiring_fn:
            wiring_fn = gate.joinpath('nidq.wiring.json')
            wiring_editor = WiringEditor()
            wiring_editor.exec_()
            wiring = wiring_editor.get_output_wiring()
            with open(wiring_fn,'w') as fid:
                json.dump(wiring,fid)
            print('Created wiring file')

    
    # RUN BACKUP
    backup.no_gui(local_run_path,remote_archive_path)

    # RUN RENAME
    ephys_data_to_alf.run(local_run_path)
    

    # Get all sessions
    sessions_paths = list(alfio.iter_sessions(local_run_path))
    sessions_paths.sort()
    for session in sessions_paths:

        # RUN PREPROCESS
        preproc_pipeline.run(session,~run_ephysQC)
        rec = Recording(session)
        rec.concatenate_alf_objects()

        # RUN SPIKESORTING
        spikeinterface_ks4.run(session,skip_remove_opto=~remove_opto_artifact)

    # Move all data:
    shutil.move(local_run_path,remote_working_path)

if __name__ == "__main__":
    main()