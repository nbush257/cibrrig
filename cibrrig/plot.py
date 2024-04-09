import matplotlib.pyplot as plt
import numpy as np

def plot_laser(intervals,amplitudes,ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    # interleave zeros for the offsets
    new_amps = np.vstack([np.zeros_like(amplitudes),amplitudes]).T.ravel()
    plt.step(intervals.ravel(),new_amps)
    