from pathlib import Path
import subprocess
import click


@click.command()
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(src_dir: Path):
    # Tarball any subdirectories of "si"
    si_dirs = list(src_dir.rglob("si"))
    for si_dir in si_dirs:
        if not si_dir.is_dir():
            continue
        print(f"Tarballing {si_dir.as_posix()}")
        tar_cmd = f"tar -czf {si_dir.as_posix()}.tar.gz -C {si_dir.parent.as_posix()} {si_dir.name} --remove-files"
        subprocess.run(tar_cmd, shell=True, check=True)

if __name__ == "__main__":
    main()