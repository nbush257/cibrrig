'''
Code to backup the raw data to a given folder
This is a "Frozen copy" - should be identical to what is acquired day of.
Much of the functionality of "Archiver" is deprecated, but retained in case it is wanted
'''

from pathlib import Path
from PyQt5.QtWidgets import QCheckBox,QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QFileDialog, QGridLayout
from PyQt5.QtCore import Qt
import spikeglx
import sys
from datetime import datetime
import shutil
import click
import subprocess
import datetime
DEFAULT_SUBJECTS_PATH =  Path(r'D:\remote_test\Subjects')
DEFAULT_SESSION_PATH  =  Path(r'D:\test\Subjects')
DEFAULT_VIDEO_DIRECTORY = Path('D:\sglx_data')
class Archiver:
    def __init__(self,keep_raw):    
        self.subject_ID = None
        self.ephys_files_local = None
        self.ephys_files_remote = None
        self.has_imec = False
        self.has_nidq = False
        self.has_video = False
        self.has_audio = False
        self.has_log = False
        self.has_notes = False
        self.has_insertions = False
        self.num_sessions = 1
        self.subjects_path = DEFAULT_SUBJECTS_PATH
        self.subject_path = None
        self.run_path = DEFAULT_SESSION_PATH 
        self.record_date = None
        self.today_path = None
        self.keep_raw=keep_raw
        self.video_files = None
        self.video_path_remote = None
        self.session_list_local = None
    
    def set_subjects_path(self, path):
        self.subjects_path = Path(path)
        print(f'Subjects path set to: {self.subjects_path}')


    def get_sessions_local(self):
        contents = self.run_path.glob('*_g[0-9]*')
        self.session_list_local = [x for x in contents if x.is_dir()]
        self.num_sessions = len(self.session_list_local)


    def guess_subject_ID(self):
        self.subject_ID = self.run_path.name
        print(f'Subject ID set to: {self.subject_ID}')

    def get_ephys_files_remote(self):
        self.ephys_files_remote = spikeglx.glob_ephys_files(self.today_path)

    def get_ephys_files_local(self):
        self.ephys_files_local = spikeglx.glob_ephys_files(self.run_path)

    def compress_ephys_files_remote(self):
        if self.ephys_files_remote is None:
            self.get_ephys_files_remote()

        def _run_compress(bin_file):
            if bin_file is None:
                return(None)
            print(f'Compressing {bin_file.name}')
            md = spikeglx.read_meta_data(bin_file.with_suffix('.meta'))
            sample_rate = md.get('imSampRate',md.get('niSampRate'))
            n_channels = int(md.get('nSavedChans'))
            compression_ratio = spikeglx.mtscomp.compress(bin_file,sample_rate=sample_rate,n_channels=n_channels,dtype='int16')
            print(f'{compression_ratio=:0.02f}')
            if not self.keep_raw:
                bin_file.unlink()
                print('removed.')

        for efi in self.ephys_files_remote:
            ap_file = efi.get('ap')
            lf_file = efi.get('lf')
            ni_file = efi.get('nidq')
            # Not using the reader object becuase it currently does not support the commercial 2.0 4 shank (version 2013)
            _run_compress(ap_file)
            _run_compress(lf_file)
            _run_compress(ni_file)
    
            
    def get_record_date(self):
        if self.ephys_files_local is None:
            self.get_ephys_files_local()
        efi = self.ephys_files_local[0]
        bin_file = efi.get('ap',efi.get('nidq'))
        md = spikeglx.read_meta_data(bin_file.with_suffix('.meta'))
        self.record_date  = md.fileCreateTime[:10] # As a string

    def make_rec_date_target(self):
        if self.record_date is None:
            self.get_record_date()
        self.subject_path = self.subjects_path.joinpath(self.subject_ID)
        self.subject_path.mkdir(exist_ok=True)
        
        self.today_path = self.subject_path.joinpath(self.record_date)
        self.today_path.mkdir(exist_ok=True)



    def copy_sessions(self):
        if self.today_path is None:
            self.make_rec_date_target()

        for session in self.session_list_local:
            dst = self.today_path.joinpath(session.name)
            print(f'Copying {session} to {dst}')
            try:
                shutil.copytree(session,dst)
            except FileExistsError:
                print(f'WARNING:Destination {dst} exists. Skipping copy!') 
    
    def copy_sessions_alf(self):
        if self.today_path is None:
            self.make_rec_date_target()

        for ii,session in enumerate(self.session_list_local):
            dst = self.today_path.joinpath(f'{ii+1:03.0f}')
            print(f'Copying {session} to {dst}')
            try:
                shutil.copytree(session,dst)
            except FileExistsError:
                print(f'WARNING:Destination {dst} exists. Skipping copy!') 
        


    
    def compress_video_in_place(self):
        self.get_videos_in_sessions()
        print(f'{self.video_files=}')
        if len(self.video_files)==0:
            print('No video files found. Continuing')
            return
        for fn in self.video_files:
            fn_comp = fn.with_suffix('.mp4')
            subprocess.run(['ffmpeg','-i',str(fn),'-y','-c:v','hevc_nvenc','-preset','slow','-cq','22',str(fn_comp)])
            fn.unlink()

    
    def get_videos_in_sessions(self):
        print(f'{self.run_path=}')
        self.video_files = list(self.run_path.rglob('*.avi'))
        if len(self.video_files)>0:
            self.has_video=True
        

        

            
    def mark_backup(self):
        for session in self.session_list_local:
            backup_flag = session.joinpath('is_backed_up.txt')
            with open(backup_flag,'w') as fid:
                fid.write(f'Archived on {datetime.datetime.today().isoformat()}')
        


class RecordingInfoUI(QWidget):
    """Ask user for the destination (Subjects path), source .run_path), and subject_ID (spikeglx run name) 

    Args:
        QWidget (_type_): _description_
    """    
    def __init__(self, archiver,title):
        super().__init__()

        self.archiver = archiver

        self.init_ui(title)
        self.resize(1200,400)

    def init_ui(self,title):

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
        main_layout.addLayout(layout,0,0)


        self.setLayout(main_layout)
        self.setWindowTitle(title)

    def browse_subjects_clicked(self):
        path = QFileDialog.getExistingDirectory(self, "Select Subjects Path (Archive destination)", str(self.archiver.subjects_path))
        if path:
            self.subjects_path_line_edit.setText(path)

    def browse_run_clicked(self):
        default_session_path = str(DEFAULT_SESSION_PATH)
        path = QFileDialog.getExistingDirectory(self, "Select Run Path (Local recording)", default_session_path)
        if path:
            print(f'run_path set to {path}')
            self.run_path_line_edit.setText(path)
            self.archiver.run_path = Path(path)
            self.archiver.guess_subject_ID()
            self.subject_id_line_edit.setText(self.archiver.subject_ID)

    def set_clicked(self):
        self.subjects_path = Path(self.subjects_path_line_edit.text())
        self.run_path = Path(self.run_path_line_edit.text())
        self.subject_id = Path(self.subject_id_line_edit.text())
        print(f'{self.subjects_path=}\n{self.run_path=}\n{self.subject_id=}')
        self.close()
    
    def open_file_dialog(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", str(DEFAULT_VIDEO_DIRECTORY), "Video Files (*.mp4 *.avi)", options=options)
        if files:
            self.video_path_line_edit.setText(", ".join(files))
        files.sort()
        self.archiver.video_files = [Path(x) for x in files]

    
    def update_has_video_state(self, state):
        # Update self.archiver.has_video when the checkbox state changes
        self.archiver.has_video = (state == Qt.Checked)


@click.command()
@click.argument('local_run_path',required=False)
@click.argument('remote_subjects_path',required=False)
@click.option('--keep_raw',is_flag=True,help='If passed does not remove the raw bin file after compression.')
def main(local_run_path,remote_subjects_path,keep_raw):
    if local_run_path is None and remote_subjects_path is None:
        archive(keep_raw)
    elif local_run_path is not None and remote_subjects_path is not None:
        no_gui(local_run_path,remote_subjects_path)
    else:
        click.echo('Invalid number of arguments')
    

def archive(keep_raw):
    '''
    Run with GUI
    '''
    app = QApplication(sys.argv)
    
    archiver = Archiver(keep_raw)
    set_path_dialog = RecordingInfoUI(archiver,"Select archival storage location for freeze (on Archive)")
    
    set_path_dialog.show()
    app.exec()
    
    archiver.get_sessions_local()
    archiver.make_rec_date_target()
    archiver.copy_sessions()
    archiver.compress_ephys_files_remote()
    archiver.compress_video_in_place()
    archiver.mark_backup()


def no_gui(local_run_path,remote_subjects_path):
    '''
    Run without gui
    '''
    archiver=Archiver(keep_raw=False)
    archiver.run_path = Path(local_run_path)
    archiver.subjects_path = Path(remote_subjects_path)
    archiver.guess_subject_ID()

    archiver.get_sessions_local()
    archiver.make_rec_date_target()
    archiver.copy_sessions()
    archiver.compress_ephys_files_remote()
    archiver.compress_video_in_place()
    archiver.mark_backup()

if __name__ == '__main__':
    main()