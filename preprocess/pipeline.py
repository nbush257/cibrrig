import subprocess
import click
import logging
from pathlib import Path
from ibllib.ephys.ephysqc import EphysQC
logging.basicConfig()
_log = logging.getLogger('PIPELINE')
_log.setLevel(logging.INFO)

def run_ephys_qc_session(session_path):
    session_path = Path(session_path)
    probe_paths = session_path.glob('raw_ephys_data/probe*')
    for probe_path in probe_paths:
        qc = EphysQC(session_path,use_alyx=False)
        qc.probe_path = probe_path
        qc.run()

@cli.command()
@click.argument('session_path', type=click.Path(exists=True))
def main(session_path):
    _log.info('RUNNING PREPROCESSING')
    command_extract_sync = ['python','extract_sync_times.py',session_path]
    command_extract_frames = ['python','extract_frame_times.py',session_path]
    command_extract_opto = ['python','extract_opto_times.py',session_path]
    command_extract_physiology = ['python','extract_physiology.py',session_path] 
    try:
        subprocess.run(command_extract_sync)
        subprocess.run(command_extract_frames)
        subprocess.run(command_extract_opto)
        subprocess.run(command_extract_physiology)
        run_ephys_qc_session(session_path)
    except:
        _log.error('Errored out')


if __name__ == '__main__':
    main()