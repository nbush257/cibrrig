import sys
sys.path.append('../')
from utils.alf_utils import Recording
from pathlib import Path
import click

@click.command()
@click.argument('session_path')
def main(session_path):
    session_path = Path(session_path)
    rec = Recording(session_path=session_path)
    rec.concatenate_alf_objects()

if __name__ == '__main__':
    main()