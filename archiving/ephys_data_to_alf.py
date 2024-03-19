'''
reformat our raw data to match IBL-like ALF standards.
This renames things in place so amke sure the raw data are backed up.

Assumes data is created by default out of spikeglx.
Runs on a 'run_path'

'''
# TODO: rename Audio

from ibllib.pipes import misc
import spikeglx
from pathlib import Path
import re
import shutil
from one.alf import spec
import click
import logging
import json
logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

DEFAULT_IMEC = {"SYSTEM": "3B","SYNC_WIRING_DIGITAL":{"P0.6": "imec_sync"}}
DEFAULT_NIDQ = {
    "SYSTEM": "3B",
    "SYNC_WIRING_DIGITAL": {
        "P0.0": "record",
        "P0.6": "right_camera",
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

def rename_probe_folders(session_path):
    raw_ephys_folder = session_path.joinpath('raw_ephys_data')
    raw_ephys_folder.mkdir(exist_ok=True)
    probe_paths = []
    def _get_probe_number(probe_path):
        probe_num = int(re.search('(?<=imec)\d',probe_path.name).group())
        out_str = f'probe{probe_num:02.0f}'
        return(out_str)

    ap_bin_files = list(session_path.rglob('*.ap.bin'))
    probe_paths_origs = set([x.parent for x in ap_bin_files])
    for probe_path_orig in probe_paths_origs:
        print(probe_path_orig)
        probe_path_dest = raw_ephys_folder.joinpath(_get_probe_number(probe_path_orig))
        probe_path_orig.rename(probe_path_dest)
        probe_paths.append(probe_path_dest)
    return probe_paths


def rename_and_move_video(session_path):
    video_files = list(session_path.rglob('*.mp4'))
    if len(video_files)==0:
        _log.warning('No video files found')
        return

    raw_video_folder = session_path.joinpath('raw_video_data')
    raw_video_folder.mkdir()
    for ii,fn in enumerate(video_files):
        prefix = fn.stem[:-20]
        alf_fn = spec.to_alf(prefix,'raw',fn.suffix,'cibrrig',extra=f'v{ii:02.0f}')
        fn_out = raw_video_folder.joinpath(alf_fn)
        _log.info(f'Renaming {fn} to {fn_out}')
        shutil.move(fn,fn_out)


def check_backup_flag(session_path):
    backup_files = session_path.joinpath('is_backed_up.txt')
    assert(backup_files.exists()),'Files have not yet been backed up. Not perfomring renaming'


def remove_backup_flag(session_path):
    try:
        backup_files = session_path.joinpath('is_backed_up.txt')
        backup_files.unlink()
    except:
        _log.warning('Issue removing backup check files')


def get_record_date(gate):
    ephys_files = spikeglx.glob_ephys_files(gate)
    if len(ephys_files)==0:
        _log.warning(f'No ephys files found in {gate}')
        return None
    efi = ephys_files[0]
    bin_file = efi.get('ap',efi.get('nidq'))
    md = spikeglx.read_meta_data(bin_file.with_suffix('.meta'))
    record_date  = md.fileCreateTime[:10] # As a string
    return record_date


def get_gate_number(gate):
    gate_num = int(re.search('(?<=_g)\d',gate.name).group())
    return gate_num


def check_wiring(session_path):
    ephys_files = spikeglx.glob_ephys_files(session_path)
    for efi in ephys_files:
        wiring_fn = list(efi['path'].glob('*wiring.json'))
        if len(wiring_fn)==0:
            if 'nidq' in efi.keys():
                prefix = 'nidq'
                sync_map = DEFAULT_NIDQ
            else:
                prefix = 'imec'
                sync_map = DEFAULT_IMEC

            wiring_fn = efi['path'].joinpath(f'{prefix}.wiring.json')
            with open(wiring_fn,'w') as fid:
                json.dump(sync_map,fid)
            _log.warning(f'No wiring.json file found. Using default map at {wiring_fn}')


def move_ni_files(session_path):
    raw_ephys_folder = session_path.joinpath('raw_ephys_data')
    raw_ephys_folder.mkdir(exist_ok=True)
    ni_files = list(session_path.glob('*nidq*'))
    for fn in ni_files:
        shutil.move(fn,raw_ephys_folder)

def rename_session(session_path):
    rename_probe_folders(session_path)
    move_ni_files(session_path)
    check_wiring(session_path)
    misc.rename_ephys_files(session_path)
    misc.move_ephys_files(session_path)
    misc.multi_parts_flags_creation(session_path)
    misc.delete_empty_folders(session_path,dry=False)
    rename_and_move_video(session_path)
    remove_backup_flag(session_path)

class Run:
    def __init__(self,run_path):
        self.run_path = run_path
        self.get_gates()
    
    def get_gates(self):
        gates = []
        # guess that we will never have more than 99 gates. This is dirty, but works
        gate_list = list(self.run_path.glob('*_g[0-9]')) + list(self.run_path.glob('*_g[0-9][0=9]'))
        for gate in gate_list:
            gates.append(gate) if gate.is_dir() else None
        gates.sort()
        self.gates = gates

    
    def move_gates(self):
        sessions = []
        for gate in self.gates:
            rec_date = get_record_date(gate)
            if rec_date is None:
                continue
            date_dir = self.run_path.joinpath(rec_date)
            date_dir.mkdir(exist_ok=True)
            gate_num = get_gate_number(gate)
            session_dir = date_dir.joinpath(f'{gate_num:03.0f}')
            gate.rename(session_dir)
            sessions.append(session_dir)
        self.sessions = sessions



@click.command()
@click.argument('run_path')
@click.option('--skip_backup_check',is_flag=True)
def main(run_path,skip_backup_check):
    run_path = Path(run_path)
    run = Run(run_path)
    if skip_backup_check:
        _log.warning('Skipping backup check!')
    else:
        for gate in run.gates:
            check_backup_flag(gate)
    run.move_gates()
    for session in run.sessions:
        rename_session(session)

if __name__ == '__main__':
    main()
