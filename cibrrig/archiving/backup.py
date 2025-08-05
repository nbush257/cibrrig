"""
This module provides functionality to backup raw electrophysiological data to a specified folder.
The backup is a "Frozen copy" that should be identical to the data acquired on the day of recording.

Much of the functionality of the `Archiver` class is deprecated but retained for legacy support.

Key features:
- Handles electrophysiological (ephys), video, audio, and metadata files.
- Compresses electrophysiological files using SpikeGLX, optionally removing the raw data after compression.
- Supports GUI-based file selection using PyQt5 for the backup process.
- Supports running the backup without a GUI through the command line.
- Copies the session data to a new target location based on the recording date.
- Marks the backup with a timestamp indicating when it was archived.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QGridLayout,
)
from PyQt5.QtCore import Qt
import spikeglx
import sys
import shutil
import click
import subprocess
import datetime
import logging

logging.basicConfig()
_log = logging.getLogger(name=__name__)
_log.setLevel(logging.INFO)
DEFAULT_SUBJECTS_PATH = Path(r"D:\remote_test\Subjects")
DEFAULT_SESSION_PATH = Path(r"D:\test\Subjects")
DEFAULT_VIDEO_DIRECTORY = Path(r"D:\sglx_data")


class Archiver:
    """
    Handles the backup of electrophysiological and related session data for a given subject.

    Attributes:
        keep_raw (bool): Whether to keep raw data after compression.
        subject_ID (str): ID of the subject being backed up.
        ephys_files_local (list): Local electrophysiological files to back up.
        ephys_files_remote (list): Remote electrophysiological files to back up.
        num_sessions (int): Number of local sessions found for the subject.
        subjects_path (Path): Path where the subjects' data is stored remotely.
        run_path (Path): Path to the local session data.
        today_path (Path): Path to the backup directory based on the record date.
        video_files (list): List of video files found in the sessions.
    """

    def __init__(self, keep_raw):
        self.subject_ID = None
        self.ephys_files_local = None
        self.ephys_files_target = None
        self.has_imec = False
        self.has_nidq = False
        self.has_video = False
        self.has_audio = False
        self.has_log = False
        self.has_notes = False
        self.has_insertions = False
        self.num_sessions = 1
        self.subjects_path_target = DEFAULT_SUBJECTS_PATH
        self.subject_path_target = None
        self.run_path_source = DEFAULT_SESSION_PATH
        self.record_date = None
        self.today_path_target = None
        self.keep_raw = keep_raw
        self.video_files = None
        self.video_path_target = None
        self.session_list_source = None

    def set_subjects_path(self, path):
        """Set the path where the subjects' data will be archived."""
        self.subjects_path_target = Path(path)
        print(f"Subjects path set to: {self.subjects_path_target}")

    def get_sessions_local(self):
        """Find local sessions to back up."""
        contents = self.run_path_source.glob("*_g[0-9]*")
        self.session_list_source = [x for x in contents if x.is_dir()]
        self.num_sessions = len(self.session_list_source)

    def guess_subject_ID(self):
        """Infer the subject ID from the local session path."""
        self.subject_ID = self.run_path_source.name
        print(f"Subject ID set to: {self.subject_ID}")

    def get_ephys_files_remote(self):
        """Retrieve remote electrophysiological files to be backed up."""
        self.ephys_files_target = spikeglx.glob_ephys_files(self.today_path_target)

    def get_ephys_files_local(self):
        """Retrieve local electrophysiological files."""
        self.ephys_files_local = spikeglx.glob_ephys_files(self.run_path_source)

    def compress_ephys_files_remote(self):
        """Compress remote electrophysiological files."""
        if self.ephys_files_target is None:
            self.get_ephys_files_remote()

        def _run_compress(bin_file):
            if bin_file is None:
                return None
            SR = spikeglx.Reader(bin_file)

            if SR.is_mtscomp:
                print(f"{bin_file.name} is already compressed")
                return None

            try:
                SR.compress_file(keep_original=self.keep_raw)
            except PermissionError:
                if ~self.keep_raw:
                    SR.close()
                    import time

                    time.sleep(1)
                    bin_file.unlink()
                    _log.warning(
                        "Compression likely did not succeed in deleting. Deleting manually."
                    )

            print(f"Compressing {bin_file.name}")

        for efi in self.ephys_files_target:
            ap_file = efi.get("ap")
            lf_file = efi.get("lf")
            ni_file = efi.get("nidq")
            # Not using the reader object becuase it currently does not support the commercial 2.0 4 shank (version 2013)
            _run_compress(ap_file)
            _run_compress(lf_file)
            _run_compress(ni_file)

    def get_record_date(self):
        """Get the recording date from the metadata of local electrophysiological files."""
        if self.ephys_files_local is None:
            self.get_ephys_files_local()
        efi = self.ephys_files_local[0]
        bin_file = efi.get("ap", efi.get("nidq"))
        md = spikeglx.read_meta_data(bin_file.with_suffix(".meta"))
        self.record_date = md.fileCreateTime[:10]  # As a string

    def make_rec_date_target(self):
        """Create the target directory for the backup, named by the record date."""
        if self.record_date is None:
            self.get_record_date()
        self.subject_path_target = self.subjects_path_target.joinpath(self.subject_ID)
        self.subject_path_target.mkdir(exist_ok=True)

        self.today_path_target = self.subject_path_target.joinpath(self.record_date)
        self.today_path_target.mkdir(exist_ok=True)

    def copy_sessions(self):
        """Copy session data from the local to the target directory."""
        if self.today_path_target is None:
            self.make_rec_date_target()

        for session in self.session_list_source:
            dst = self.today_path_target.joinpath(session.name)
            print(f"Copying {session} to {dst}")
            try:
                shutil.copytree(session, dst)
            except FileExistsError:
                print(f"WARNING:Destination {dst} exists. Skipping copy!")

    def copy_sessions_alf(self):
        """Compress any video files in the sessions."""
        if self.today_path_target is None:
            self.make_rec_date_target()

        for ii, session in enumerate(self.session_list_source):
            dst = self.today_path_target.joinpath(f"{ii+1:03.0f}")
            print(f"Copying {session} to {dst}")
            try:
                shutil.copytree(session, dst)
            except FileExistsError:
                print(f"WARNING:Destination {dst} exists. Skipping copy!")

    def compress_video_in_place(self):
        """Compress any video files in the sessions."""
        self.get_videos_in_sessions()
        print(f"{self.video_files=}")
        if len(self.video_files) == 0:
            print("No uncompressed video files found. Continuing")
            return
        for fn in self.video_files:
            fn_comp = fn.with_suffix(".mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(fn),
                    "-y",
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    "slow",
                    "-cq",
                    "22",
                    str(fn_comp),
                ]
            )
            fn.unlink()

    def get_videos_in_sessions(self):
        """Find video files in the local session directories."""
        print(f"{self.run_path_source=}")
        self.video_files = list(self.run_path_source.rglob("*.avi"))
        if len(self.video_files) > 0:
            self.has_video = True

    def mark_backup(self):
        """Create a flag file in each session directory to indicate the backup date."""
        for session in self.session_list_source:
            backup_flag = session.joinpath("is_backed_up.txt")
            with open(backup_flag, "w") as fid:
                fid.write(f"Archived on {datetime.datetime.today().isoformat()}")

    def copy_run_level_files(self):
        """ Copy files in the run folder to the backup location"""
        run_files = list(self.run_path_source.glob("*"))
        # Remove directories from the list
        run_files = [x for x in run_files if x.is_file()]
        print(f"{run_files=}")
        for fn in run_files:
            shutil.copy(fn, self.today_path_target)
    
    def full_archive(self):
        """ Chain all the methods to perform a full archive"""
        self.get_sessions_local()
        self.make_rec_date_target()
        self.copy_run_level_files()
        self.copy_sessions()
        self.compress_ephys_files_remote()
        self.compress_video_in_place()
        self.mark_backup()


class RecordingInfoUI(QWidget):
    """
    GUI to ask the user for the destination (Subjects path), source (run_path), and subject_ID (spikeglx run name).

    Attributes:
        archiver (Archiver): Instance of the `Archiver` class to manage the backup.
    """

    def __init__(self, archiver, title):
        super().__init__()

        self.archiver = archiver

        self.init_ui(title)
        self.resize(1200, 400)

    def init_ui(self, title):
        """Set up the UI layout and widgets."""
        main_layout = QGridLayout()

        layout = QVBoxLayout()

        label_subjects_path = QLabel("Select Subjects Path:")
        self.subjects_path_line_edit = QLineEdit()
        self.subjects_path_line_edit.setText(str(self.archiver.subjects_path))
        browse_subjects_button = QPushButton("Browse")
        browse_subjects_button.clicked.connect(self.browse_subjects_clicked)

        label_run_path = QLabel("Select Run Path:")
        self.run_path_line_edit = QLineEdit()
        self.run_path_line_edit.setText(str(self.archiver.run_path))
        browse_run_button = QPushButton("Browse")
        browse_run_button.clicked.connect(self.browse_run_clicked)

        label_subject_id = QLabel("Enter Subject ID:")
        self.subject_id_line_edit = QLineEdit()

        set_button = QPushButton("Set Paths")
        set_button.clicked.connect(self.set_clicked)

        layout.addWidget(label_subjects_path)
        layout.addWidget(self.subjects_path_line_edit)
        layout.addWidget(browse_subjects_button)

        layout.addWidget(label_run_path)
        layout.addWidget(self.run_path_line_edit)
        layout.addWidget(browse_run_button)

        layout.addWidget(label_subject_id)
        layout.addWidget(self.subject_id_line_edit)

        layout.addWidget(set_button)
        main_layout.addLayout(layout, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle(title)

    def browse_subjects_clicked(self):
        """Open a dialog for the user to select the subjects path."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Subjects Path (Archive destination)",
            str(self.archiver.subjects_path),
        )
        if path:
            self.subjects_path_line_edit.setText(path)

    def browse_run_clicked(self):
        """Open a dialog for the user to select the local run path."""
        default_session_path = str(DEFAULT_SESSION_PATH)
        path = QFileDialog.getExistingDirectory(
            self, "Select Run Path (Local recording)", default_session_path
        )
        if path:
            print(f"run_path set to {path}")
            self.run_path_line_edit.setText(path)
            self.archiver.run_path = Path(path)
            self.archiver.guess_subject_ID()
            self.subject_id_line_edit.setText(self.archiver.subject_ID)

    def set_clicked(self):
        """Set the paths and subject ID from the user input."""
        self.subjects_path = Path(self.subjects_path_line_edit.text())
        self.run_path = Path(self.run_path_line_edit.text())
        self.subject_id = Path(self.subject_id_line_edit.text())
        print(f"{self.subjects_path=}\n{self.run_path=}\n{self.subject_id=}")
        self.close()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            str(DEFAULT_VIDEO_DIRECTORY),
            "Video Files (*.mp4 *.avi)",
            options=options,
        )
        if files:
            self.video_path_line_edit.setText(", ".join(files))
        files.sort()
        self.archiver.video_files = [Path(x) for x in files]

    def update_has_video_state(self, state):
        # Update self.archiver.has_video when the checkbox state changes
        self.archiver.has_video = state == Qt.Checked


@click.command()
@click.argument("local_run_path", required=False)
@click.argument("remote_subjects_path", required=False)
@click.option(
    "--keep_raw",
    is_flag=True,
    help="If passed does not remove the raw bin file after compression.",
)
def main(local_run_path, remote_subjects_path, keep_raw):
    """
    Entry point for the backup process.

    If no arguments are provided, the GUI will open.
    If both `local_run_path` and `remote_subjects_path` are provided, the process will run without the GUI.

    Args:
        local_run_path (str): Path to the local recording session.
        remote_subjects_path (str): Path to the remote storage location.
        keep_raw (bool): Flag to indicate whether raw data should be kept after compression.
    """
    if local_run_path is None and remote_subjects_path is None:
        archive(keep_raw)
    elif local_run_path is not None and remote_subjects_path is not None:
        no_gui(local_run_path, remote_subjects_path)
    else:
        click.echo("Invalid number of arguments")


def archive(keep_raw):
    """
    Run the backup process with a GUI.

    Args:
        keep_raw (bool): Whether to keep raw data after compression.
    """
    app = QApplication(sys.argv)

    archiver = Archiver(keep_raw)
    set_path_dialog = RecordingInfoUI(
        archiver, "Select archival storage location for freeze (on Archive)"
    )

    set_path_dialog.show()
    app.exec()

    archiver.full_archive()


def no_gui(local_run_path, remote_subjects_path):
    """
    Run the backup process without a GUI, using command line arguments.

    Args:
        local_run_path (str): Path to the local recording session.
        remote_subjects_path (str): Path to the remote storage location.
    """
    archiver = Archiver(keep_raw=False)
    archiver.run_path_source = Path(local_run_path)
    archiver.subjects_path_target = Path(remote_subjects_path)
    archiver.guess_subject_ID()

    archiver.full_archive()


if __name__ == "__main__":
    main()
