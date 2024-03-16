import subprocess
import click
import logging
from pathlib import Path
logging.basicConfig()
_log = logging.getLogger('PIPELINE')
_log.setLevel(logging.INFO)

from ibllib.ephys.ephysqc import EphysQC
#TODO: eventually want to have the extract physiology read the wiring json. Bigger project.

def run_ephys_qc_session(session_path):
    session_path = Path(session_path)
    probe_paths = session_path.glob('raw_ephys_data/probe*')
    for probe_path in probe_paths:
        qc = EphysQC(session_path,use_alyx=False)
        qc.probe_path = probe_path
        # NEB made an edit to "spikeglx" that is not consistent with teh current version from the IBL that allows the 2013 version (commercial np2.0 5 shank probes)
        qc.run()


@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command()
@click.argument('session_path', type=click.Path(exists=True))
@click.pass_context
def awake(ctx,session_path):
    _log.info('RUNNING AWAKE PREPROCESSING')
    command_extract_sync = ['python','extract_sync_times.py',session_path]
    command_extract_frames = ['python','extract_frame_times.py',session_path]
    command_extract_opto = ['python','extract_opto_times.py',session_path]
    command_extract_physiology = ['python','extract_physiology.py',session_path,
                                  '-d','-1','-e','-1','-t','-1'] # Many of the physiology is not kept

    try:
        subprocess.run(command_extract_sync)
        subprocess.run(command_extract_frames)
        subprocess.run(command_extract_opto)
        subprocess.run(command_extract_physiology)
        run_ephys_qc_session(session_path)
    except:
        _log.error('Errored out')

@cli.command()
@click.argument('session_path', type=click.Path(exists=True))
@click.pass_context
def anesthetized(ctx,session_path):
    _log.info('RUNNING ANESTHETIZED PREPROCESSING')
    command_extract_sync = ['python','extract_sync_times.py',session_path]
    command_extract_opto = ['python','extract_opto_times.py',session_path]
    command_extract_physiology = ['python','extract_physiology.py',session_path] 

    try:
        subprocess.run(command_extract_sync)
        subprocess.run(command_extract_opto)
        subprocess.run(command_extract_physiology)
        run_ephys_qc_session(session_path)
    except:
        _log.error('Errored out')



if __name__ == '__main__':
    cli(obj={})