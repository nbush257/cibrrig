import subprocess
import click
import logging
from ibllib.ephys.ephysqc import extract_rmsmap
import spikeglx
import sys
logging.basicConfig()
_log = logging.getLogger('PIPELINE')
_log.setLevel(logging.INFO)


def run_ephys_qc_session(session_path):
    """Get the RMS maps only on the longest recording of a session

    Args:
        session_path (_type_): _description_

    Returns:
        _type_: _description_
    """    
    efiles = spikeglx.glob_ephys_files(session_path)
    max_duration = 0 
    qc_files = []
    longest_ap = None
    longest_lf = None
    for efile in efiles:
        if efile.get('ap') and efile.ap.exists():
            SR = spikeglx.Reader(efile.ap)
            if SR.ns>max_duration:
                max_duration = SR.ns
                longest_ap = efile['ap']
                if efile.get('lf'):
                    longest_lf = efile['lf']
    if longest_ap:
        SR = spikeglx.Reader(longest_ap)
        qc_files.extend(extract_rmsmap(SR, out_folder=None, overwrite=False))
    if longest_lf:
        SR = spikeglx.Reader(longest_lf)
        qc_files.extend(extract_rmsmap(SR, out_folder=None, overwrite=False))
    return qc_files


@click.command()
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