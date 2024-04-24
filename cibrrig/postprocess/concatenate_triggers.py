try:
    from ..utils.alf_utils import Recording
except:
    import sys
    sys.path.append('../')
    from utils.alf_utils import Recording
from pathlib import Path
import click

@click.command()
@click.argument('session_path')
@click.option('--skip_overwrite',is_flag=True)
def main(session_path,skip_overwrite):
    session_path = Path(session_path)
    rec = Recording(session_path=session_path)
    rec.concatenate_alf_objects(overwrite=~skip_overwrite)

if __name__ == '__main__':
    main()