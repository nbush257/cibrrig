"""
CLI to concatenate all the triggers in a session.
Probably should be re-organized with the alfutils maodule
"""

from cibrrig.utils.alf_utils import Recording
from pathlib import Path
import click


@click.command()
@click.argument("session_path")
@click.option("--skip_overwrite", is_flag=True)
def main(session_path, skip_overwrite):
    session_path = Path(session_path)
    rec = Recording(session_path=session_path)
    rec.concatenate_session(overwrite=~skip_overwrite)


if __name__ == "__main__":
    main()
