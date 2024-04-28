import subprocess
import click
import logging
from ibllib.ephys.ephysqc import extract_rmsmap,EphysQC
import spikeglx
import sys
from pathlib import Path
from . import extract_frame_times,extract_opto_times,extract_physiology,extract_sync_times
logging.basicConfig()
_log = logging.getLogger('PIPELINE')
_log.setLevel(logging.INFO)


def DEPRECATED_run_ephys_qc_session(session_path):
    """(DEPRECATED - VERY SLOW)Get the RMS maps only on the longest recording of a session

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

def run_ephys_qc_session(session_path):
    session_path = Path(session_path)
    probe_paths = list(session_path.glob('raw_ephys_data/probe[0-9][0-9]'))
    for probe_path in probe_paths:
        qc = EphysQC(probe_id=probe_path.name,session_path=session_path,use_alyx=False)
        qc.probe_path = probe_path
        qc.run()


@click.command()
@click.argument('session_path', type=click.Path(exists=True))
@click.option('--skip_ephysqc',is_flag=True)
def main(session_path,skip_ephysqc):
    _log.info('RUNNING PREPROCESSING')
    _log.info('Skipping ephysQC') if skip_ephysqc else None
    try:
        extract_sync_times.main(session_path)
        extract_frame_times.main(session_path)
        extract_opto_times.main(session_path)
        extract_physiology.main(session_path)
        if not skip_ephysqc:
            run_ephys_qc_session(session_path)
    except:
        _log.error('Errored out')


if __name__ == '__main__':
    main()