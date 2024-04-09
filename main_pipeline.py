'''
Takes a spikeglx run and performs:
1) compression and copy backup to archive 
2) Renaming to alf
3) Preprocessing
4) Spikesorting

# Gui created by chatgpt
'''
from pathlib import Path
import subprocess
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QCheckBox, QGridLayout
from PyQt5.QtCore import Qt
from pathlib import Path
import one.alf.io as alfio
import shutil
import os
from utils.alf_utils import Recording

class DirectorySelector(QWidget):
    def __init__(self):
        super().__init__()
        self.local_run_path = Path("D:/sglx_data/Subjects")
        self.remote_archive_path = Path("Z:/projects")
        self.remote_working_path = Path("Y:/projects")
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

def main():
    app = QApplication(sys.argv)
    window = DirectorySelector()
    window.show()
    app.exec_()

    # After the GUI is closed, retrieve the selected paths
    local_run_path, remote_archive_path, remote_working_path, remove_opto_artifact,run_ephysQC = window.get_paths()

    cmd_archive = ['python','./archiving/backup.py',str(local_run_path),str(remote_archive_path)]
    cmd_rename = ['python','./archiving/ephys_data_to_alf.py',str(local_run_path)]

    gate_paths = list(local_run_path.glob('*_g[0-9]*'))
    for gate in gate_paths:
        wiring_fn = list(gate.glob('nidq.wiring.json'))
        assert wiring_fn,f'No wiring file found for {gate}. Create one before continuing'

    
    subprocess.run(cmd_archive,check=True)
    subprocess.run(cmd_rename,check=True)

    # Get all sessions
    sessions_paths = list(alfio.iter_sessions(local_run_path))
    sessions_paths.sort()
    for session in sessions_paths:
        if not run_ephysQC:
            preproc_cmd = ['python','preproc_pipeline.py',str(session),'--skip_ephysQC']
        else:
            preproc_cmd = ['python','preproc_pipeline.py',str(session)]
    
        if not remove_opto_artifact:
            sort_cmd = ['python','-W','ignore','spikeinterface_ks4.py',str(session),'--skip_remove_opto']
        else:
            sort_cmd = ['python','-W','ignore' ,'spikeinterface_ks4.py',str(session)]

        subprocess.run(preproc_cmd,check=True,cwd='./preprocess')
        rec = Recording(session)
        rec.concatenate_alf_objects()
        subprocess.run(sort_cmd,check=True,cwd='./sorting')

    # Move all data:
    shutil.move(local_run_path,remote_working_path)

if __name__ == "__main__":
    main()