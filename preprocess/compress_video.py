'''
This convinience script usses ffmpeg and the gpu to compress a video in both lossless
and lossy hevc encoding. 

'''
import subprocess
from pathlib import Path
import shutil
import click 


@click.command()
@click.argument('fn')
@click.option('--skip_lossless',is_flag=True)
@click.option('--skip_lossy',is_flag=True)
@click.option('--no_delete',is_flag=True,help = 'Do not delete the original raw uncompressed file')
def main(fn,skip_lossless,skip_lossy,no_delete):

    if skip_lossy and skip_lossless:
        print('You have chosen to skip both encodings so we are exiting before we accidentally delete raw data.')
        return(None)
    fn = Path(fn)
    if fn.suffix=='.mp4':
        raise NotImplementedError(f'Original video file suffix {fn.suffix} cannot be .mp4 as that is the target filetype')
   
    fn_lossless = fn.with_suffix('.mp4')
    fn_lossy = fn.with_suffix('.lossy.mp4')

    # Lossless encoding
    if skip_lossless:
        print('skipping lossless encoding...')
    else:
        subprocess.run(['ffmpeg','-i',str(fn),'-c:v','hevc_nvenc','-preset','default','-tune','lossless',str(fn_lossless)])

    # Lossy encoding (high quality)
    if skip_lossy:
        print('Skipping lossy encoding')
    else:
        subprocess.run(['ffmpeg','-i',str(fn),'-c:v','hevc_nvenc','-preset','slow','-cq','22',str(fn_lossy)])

    # Remove original file
    if no_delete:
        print(f'Original file kept at {fn}')
    else:
        print(f'Removing original data file {fn}')
        fn.unlink()
        
if __name__=='__main__':
    main()