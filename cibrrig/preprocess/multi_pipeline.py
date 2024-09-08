"""
DEPRECATED
Run the preprocess pipeline on every session in a Subjects folder
"""

from pathlib import Path
import subprocess
import click
import logging
import matplotlib
import sys

logging.basicConfig()
_log = logging.getLogger("multi_preproc")
_log.setLevel(logging.DEBUG)

if sys.platform == "linux":
    _log.info("HPC detected - using TkAgg")
    matplotlib.use("TkAgg")

raise NotImplementedError("This script is deprecated. Use preproc_pipeline.py instead.")


@click.command()
@click.argument("subjects_path")
def main(subjects_path):
    subjects_path = Path(subjects_path)
    sessions_paths = list(subjects_path.glob("*/*/0*/"))
    sessions_paths.sort()
    _log.debug("Sessions paths\n\t" + "\n\t".join([str(x) for x in sessions_paths]))
    for session in sessions_paths:
        _log.info(f"Preprocessing on {session}")
        command = ["python", "pipeline.py", str(session)]
        try:
            subprocess.run(command, check=True)
        except Exception:
            _log.error(f"Failure on {session}")


if __name__ == "__main__":
    main()
