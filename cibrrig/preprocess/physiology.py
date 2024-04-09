'''
Code to process physiological signals like EMGs and EKGs
Only does processing. No I/O

'''
import sklearn
import numpy as np
import pandas as pd
import warnings
from sklearn.mixture import BayesianGaussianMixture
import scipy.signal
import scipy.interpolate

def burst_stats_dia(integrated,sr,dia_thresh=1,rel_height=0.9,min_distance=0.1,min_width = 0.025):
    """
    Calculate diaphragm burst features
    Args:
        integrated (_type_): integrated diaphragm trace
        sr (_type_): sample rate of the integrated diaphragm trace
        dia_thresh (int, optional): prominence threshold (zscore). Defaults to 1.
        rel_height (float, optional): Rerlative height to use when finding onset times. Defaults to 0.9.
        min_distance (float, optional): minimum time between breaths in seconds. Effectively sets the maximum breathing rate. Defaults to 0.1.
        min_width (float, optional): minimum time duration to consider a breath in seconds. Defaults to 0.025.
    """

    '''
    :param integrated: integrated diaphragm trace
    :param sr: sample rate
    :param dia_thresh:
    :return:
            dia_df - dataframe of diaphragm burst features
    '''
    scl = sklearn.preprocessing.StandardScaler(with_mean=0)
    integrated_scl = scl.fit_transform(integrated[:,np.newaxis]).ravel()

    pks = scipy.signal.find_peaks(integrated_scl,
                                  prominence=dia_thresh,
                                  distance=int(min_distance*sr),
                                  width=int(min_width*sr))[0]
    lips = scipy.signal.peak_widths(integrated,pks,rel_height=rel_height)[2]
    rips = scipy.signal.peak_widths(integrated,pks,rel_height=rel_height)[3]
    lips = lips.astype('int')
    rips = rips.astype('int')


    #Remove onsets that are equal to offsets
    to_keep =rips!=lips
    pks = pks[to_keep]
    lips = lips[to_keep]
    rips = rips[to_keep]

    # Find breaths fully contained in other breaths (likely sighs)
    idx = np.where(np.diff(lips)<0)[0]
    to_rm = idx[rips[idx+1]>rips[idx]]
    pks = np.delete(pks,to_rm)
    lips = np.delete(lips,to_rm)
    rips = np.delete(rips,to_rm)

    # Find overlapping breaths # Keep longer
    postBI = lips[1:]-rips[:-1]
    idx = np.where(postBI<0)[0]
    dur_first = rips[idx]-lips[idx]
    dur_second = rips[idx+1]-lips[idx+1]
    temp = np.vstack([dur_first,dur_second]).T
    to_rm = idx + np.argmin(temp,1)
    pks = np.delete(pks,to_rm)
    lips = np.delete(lips,to_rm)
    rips = np.delete(rips,to_rm)
    


    amp = np.zeros(len(lips))
    auc = np.zeros(len(lips))
    for ii,(lip,rip) in enumerate(zip(lips,rips)):
        temp = integrated[lip:rip]
        amp[ii] = np.percentile(temp,95)
        auc[ii] = np.trapz(temp)
    dur = rips-lips

    lips_t = lips/sr
    rips_t = rips/sr

    dia_data = {}
    dia_data['on_samp'] = lips
    dia_data['off_samp'] = rips
    dia_data['on_sec'] = lips_t
    dia_data['off_sec'] = rips_t
    dia_data['amp'] = amp
    dia_data['auc'] = auc
    dia_data['duration_sec'] = dur/sr
    dia_data['duration_samp'] = dur
    dia_data['pk_samp'] = pks
    dia_data['pk_time'] = pks/sr
    dia_data['postBI'] = np.hstack([lips_t[1:]-rips_t[:-1],[np.nan]])
    dia_df = pd.DataFrame(dia_data)
    dia_df = dia_df.eval('inst_freq=1/(duration_sec+postBI)')
    dia_df = dia_df.eval('IBI=duration_sec+postBI')

    dia_df = dia_df.query('inst_freq>0')

    return(dia_df)


def remove_EKG(x,sr,thresh=2):
    """Remove the EKG from an ephys trace using a BGM classifier
    Implements a [5,500] bandpass filter to try to isolate the ekg

    Args:
        x (1D numpy array): ephys trace to clean (diaphragm or other emg)
        sr (float): sampling rate (samples/sec)
        thresh (int, optional): threshold in standard deviations to detect a heartbeat. Defaults to 2.

    Returns: 
            y (1D numpy array): x with the EKG filtered out. Has the same timestamps as x
            pks (1D numpy array): samples of the detected  heartbeats 
    """    

    warnings.filterwarnings('ignore')
    sos = scipy.signal.butter(2,[5/sr/2,500/sr/2],btype='bandpass',output='sos')
    xs = scipy.signal.sosfilt(sos,x)
    pks = scipy.signal.find_peaks(xs,prominence=thresh*np.std(xs),distance=0.05*sr)[0]
    pks = pks[1:-1]
    amps = xs[pks]
    win = int(0.010 *sr)
    y = x.copy()
    ekg = np.zeros([2*win,len(pks)])
    for ii,pk in enumerate(pks):
        try:
            ekg[:,ii] = x[pk-win:pk+win]
        except:
            pass

    ekg_std = np.std(ekg[:30],0) +np.std(ekg[-30:],0)
    ekg_std = np.log(ekg_std)
    mask = np.logical_not(np.isfinite(ekg_std))
    ekg_std[mask] = np.nanmedian(ekg_std)


    bgm = BayesianGaussianMixture(n_components=2)
    cls = bgm.fit_predict(ekg_std[:,np.newaxis])
    cls[cls==0]=-1
    m0 = np.nanmean(ekg_std[cls==-1])
    m1 = np.nanmean(ekg_std[cls==1])
    if m0>m1:
        cls = -cls

    ww = int(.0005 * sr)
    ww += ww % 2 - 1

    for ii,pk in enumerate(pks):
        if pk-win<0:
            continue
        if (pk+win)>len(y):
            continue
        if cls[ii]==-1:
            sm_ekg = scipy.signal.savgol_filter(ekg[:,ii],ww,1)
            y[pk - win:pk + win] -= sm_ekg
        else:
            med_ekg = np.nanmedian(ekg[:,ii-5:ii+5],1)
            med_amp = np.median(amps[ii-5:ii+5])
            scl = amps[ii]/med_amp
            y[pk - win:pk + win] -=med_ekg*scl

    y[np.isnan(y)] = np.nanmedian(y)
    warnings.filterwarnings('default')
    return(y,pks)


def get_hr_from_dia(pks,dia_df,sr):
    """
    Computes the average heart rate from the diaphragm
    Not as good as getting it from a dedicated channel

    Args:
        pks (_type_): _description_
        dia_df (_type_): _description_
        sr (_type_): _description_
    """
    ons = dia_df['on_samp']
    offs = dia_df['off_samp']
    for on,off in zip(ons,offs):
        mask = np.logical_not(
            np.logical_and(
                pks>on,
                pks<off
            )
        )
        pks = pks[mask]

    pulse = pd.DataFrame()
    pulse['hr (bpm)'] = 60*sr/np.diff(pks)
    hr_smooth = pulse.rolling(50,center=True).median()
    hr_smooth.interpolate(limit_direction='both',inplace=True)
    hr_smooth['t']=pks[:-1]/sr
    return(hr_smooth,pks/sr)


def extract_hr_from_ekg(x,sr,thresh=5,min_distance = 0.05,low=100,high=1000,in_samples= False):
    """
    Finds heartbeats in an EKG trace. 
    Uses a bandpass filter and peak detection

    Args:
        x (1D numpy array): EKG trace
        sr (float): sampling rate (smaples/sec)
        thresh (int, optional): Prominence threshold in standard deviations to identify peaks. Defaults to 5.
        min_distance (float, optional): Minimum time between detected heartbeats. Effectively sets a maximum heart rate. Defaults to 0.05.
        low (int, optional): Low cut frequency in the bandpass filter (Hz). Defaults to 100.
        high (int, optional): High cut frequency in the bandpass filter (Hz). Defaults to 1000.
        raw_pks(bool,optional): whether to return the peaks as samples. Defualts to False
    """    
    order = 8
    sos = scipy.signal.butter(order,[low/sr/2,high/sr/2],btype='bandpass',output='sos')
    xs = scipy.signal.sosfilt(sos,x)
    pks = scipy.signal.find_peaks(xs,prominence=thresh*np.std(xs),distance=int(min_distance*sr))[0]
    if in_samples:
        return(pks)
    return(pks/sr)


def compute_avg_hr(heartbeats,smoothing_window='10s',dt=0.1,t_target=None):
    """Get the mean heart rate using a sliding average window.
    First performs a median filter to remove large instantaneous outliers
    Then performs a mean filter to smooth out the average.
    Uses pandas.
    
    Can pass t_target to interpolate to a new time basis. Otherwise defualts to a 100ms window

    If both t_target and dt are None - outputs the smoothed heart rate only with length (# heartbeats)

    Args:
        heartbeats (1D numpy array): timestamps of each heartbeat. Must be in seconds
        smoothing_window (str, optional): Window to pass to pandas rolling average. Defaults to 10s.
        dt (float,optional): if t_target is not supplied, set the time step.
        t_target (1D numpy array, optional): Time vector to interpolate the heart rate to. Defaults to None. If none, picks a regularly spaced time with timestep "dt"

    Returns:
        t_target(1D numpy array): timestamps corresponding to each sample of the smoothed heart rate
        smoothed_hr_out (1D numpy array): Windowed average heart rate

    Created with the input of chatgpt 3.5
    """    
    if t_target is None:
        if dt is None:
            pass
        else:
            t_target = np.arange(0,heartbeats.max(),dt)

    # Convert heartbeat times to DatetimeIndex
    heartbeat_df = pd.DataFrame({'heartbeat_times': heartbeats})
    heartbeat_df['heartbeat_times'] = pd.to_datetime(heartbeat_df['heartbeat_times'],unit='s')

    # Calculate inter-beat intervals
    heartbeat_df['inter_beat_interval'] = heartbeat_df['heartbeat_times'].diff()

    # Convert inter-beat intervals to heart rate (beats per minute)
    heartbeat_df['heart_rate'] = 60 / heartbeat_df['inter_beat_interval'].dt.total_seconds()
    heartbeat_df.set_index('heartbeat_times',inplace=True)
    heartbeat_df['smoothed_heart_rate'] = heartbeat_df['heart_rate'].rolling(window='1s',center=True).median().rolling(smoothing_window,center=True).mean()

    if t_target is None:
        return(heartbeat_df['smoothed_heart_rate'].values)

    f = scipy.interpolate.interp1d(heartbeats,heartbeat_df['smoothed_heart_rate'].values,fill_value='extrapolate')
    smoothed_hr_out = f(t_target)

    return(t_target,smoothed_hr_out)


def compute_sighs(breath_times ,auc, thresh=7,win='20s'):
    """Use a rolling Median Absolute Deviance (MAD) to identify sighs based on the AUC of the diaphragm

    Args:
        breath_times (_type_): times of each breath
        auc (_type_): Area under the curve of the diaphragm
        thresh (int, optional): Multiplier for the MAD. Defaults to 7.
        win (str, optional): Window to estimate the rolling AUC. Defaults to '20s'.
    """    

    df = pd.DataFrame()
    df['x'] = auc
    breath_times = pd.to_datetime(breath_times,unit='s')
    df.index = breath_times

    filt_breaths = df-df.rolling(win).median()
    MAD = lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
    rolling_MAD = df.rolling(window=win, center=True).apply(MAD)*thresh

    idx = filt_breaths['x']>rolling_MAD['x']

    is_sigh = idx.values
    return(is_sigh)


def compute_onstructive_apneas(breath_times, inhale_onsets):
    """
    Convinience function that says anywhere we don't have a detected ihale onset 
    but we do have a detected breath, that is an obstructive apnea

    Args:
        breath_times (1D numpy array): times of breath onsets (not used, but required of the user so we don't accidentally process without diaphragm data)
        inhale_onsets (1D numpy array): times of inhale onsets

    Returns:
        is_obstructive apnea: - boolean true when obstructive apnea is deteected 
    """    
    is_obstructive_apnea = np.isnan(inhale_onsets)
    return is_obstructive_apnea
