from sklearn.decomposition import PCA
import numpy as np
from iblutil.numerical import bincount2D
from scipy.ndimage import gaussian_filter1d
import logging
from ..plot import plot_projection
logging.basicConfig()
_log = logging.getLogger('population')
_log.setLevel(logging.INFO)


#TODO: get phi_binned average projection (may be better in a utils folder)
def rasterize(spike_times,spike_clusters,binsize,FR=False):
    """
    Turn spike times and cluster ids into a 2D array of spikecounts per time bin
    Works on subsets of clusters (i.e., does not require the clusters to be continuous 0,1,2,3...)

    Args:
        binsize (float): Raster time bin size in seconds
    """        
    '''
    '''
    raster, tbins, cbins = bincount2D(spike_times, spike_clusters, binsize)
    raster = raster.astype('float')
    # convert spike counts to spikes per second
    if FR:
        raster = raster/binsize 
    return(raster,tbins,cbins)


def smooth_raster(raster,binsize,sigma):
    """
    Smooth a computed raster.
    Raster should be [cells x time]

    Args:
        binsize (float): 
        sigma (_type_): _description_
    """        
    sigma_scaled = sigma/binsize
    raster_smoothed = gaussian_filter1d(raster, sigma=sigma_scaled, axis=1)
    raster_smoothed[np.isnan(raster_smoothed)] = 0
    return(raster_smoothed)
    

def scale_raster(raster,transform='sqrt'):
    if transform=='sqrt':
        raster_scaled = np.sqrt(raster)
        raster_scaled[np.isinf(raster_scaled)] = 0
    else:
        raster_scaled = raster
    return(raster_scaled)


def preprocess_pca(spike_times,spike_clusters,binsize,sigma,transform='sqrt'):
    """
    wraps the preprocessing steps

    Args:
        binsize (_type_): _description_
        sigma (_type_): _description_
        transform (str, optional): _description_. Defaults to 'sqrt'.
    """        
    raster,tbins,cbins = rasterize(spike_times,spike_clusters,binsize)
    raster_smoothed = smooth_raster(raster, binsize=binsize,sigma=sigma)
    raster_scaled = scale_raster(raster_smoothed,transform=transform)
    rez  = dict(
        raster = raster,
        raster_smoothed = raster_smoothed,
        raster_scaled = raster_scaled,
        tbins = tbins,
        cbins = cbins,
    )
    return(rez)


def compute_projection_speed(X,ndims = 3):
    '''
    Compute the euclidean speed through PCA space
    :param X: PCA decompositions (2D numpy array: N_timepoints x N_dims)
    :param n: number of dimensions to use (int)
    :return: X_speed - 1D numpy array of PCA speed
    '''

    if X.shape[0]<=X.shape[1]:
        raise Warning(f'Number of timepoints:{X.shape[0]} is fewer than number of dimensions:{X.shape[1]}. Confirm you do not need to transpose the matrix')
    X_subset = X[:,:ndims]
    X_sd = np.diff(X_subset,axis=0)
    X_speed = np.concatenate([[0], np.sqrt(np.sum(X_sd ** 2, axis=1))])

    return(X_speed)


def project_pca(raster,tbins,ndims=20,t0=None,tf=None):
    '''
    Project the spiking data into the PCA space. Will fit on the interval between 
    (t0,tf), and apply to the entire spiking data 

    Raster should have shape [n_cells x time]. It will be transposed at fitting time
    
    If t0,tf is an array, fits on the intervals [t01,tf1],[t02,tf2]...
    If t0,tf is None, fits on all data
    '''
    raster_train,tbins_train =  _subset_raster(raster,tbins,t0,tf)
    pca = PCA(ndims).fit(raster_train.T)
    projected = pca.transform(raster.T)
    return(projected,pca)


def _subset_raster(raster,tbins,t0,tf):
    if t0 is None:
        t0 = np.array([np.min(tbins)])
    if tf is None:
        tf = np.array([np.max(tbins)])
    
    t0 = np.atleast_1d(t0)
    tf = np.atleast_1d(tf)
    raster_out = []
    tbins_out = []

    if len(t0)>1:
        _validate_intervals(t0,tf)

    for start,stop in zip(t0,tf):
        s0,sf = np.searchsorted(tbins, [start,stop])
        raster_out.append(raster[:,s0:sf])
        tbins_out.append(tbins[s0:sf])
    raster_out = np.hstack(raster_out)
    tbins_out = np.concatenate(tbins_out)

    return(raster_out,tbins_out)


# May be better in a utils file
def _validate_intervals(starts,stops):
    '''
    Validates that two vectors are indeed intervals (monotonic, causal, and non-overlapping)
    '''
    assert np.all(np.diff(starts)>0),'Starts is not monotonic'
    assert len(starts)==len(stops),f'Number of starts {len(starts)} does not match number of stops {len(stops)}'
    assert np.all(stops>=starts), 'Stops are not all after starts'
    assert np.all(starts[1:]>=stops[:-1]), 'Intervals are overlapping'


# May be better in a utils file
def remap_time_basis(x,x_t,y_t):
    '''
    Convinience function to map an analog signal x into the time
    basis for another signal y.
    ex: x is phase, y is the PCA decomposition. This allows you to get the phase value for
    each sample in the PCA time
    :param x: Analog signal to change time basis (1D numpy array)
    :param x_t: Time basis of original analog signal (1D numpy array)
    :param y_t: Time basis of target signal (1D numpy array)
    :return: x_mapped - x in the time basis of y (1D numpy array)
    '''
    assert(len(x)==len(x_t))
    idx = np.searchsorted(x_t,y_t)-1
    assert(len(idx)==len(y_t))
    return(x[idx])


class Population:
    def __init__(self,spike_times,spike_clusters,
                 ndims=None,binsize=0.005,sigma=0.01,t0=None,tf=None):
        self.spike_times = spike_times
        self.spike_clusters = spike_clusters
        self.ndims = ndims
        self.binsize = binsize
        self.sigma = sigma
        self.t0 = t0
        self.tf = tf
        self.raster = None
        self.raster_smoothed = None
        self.cbins = None
        self.tbins = None
        self.projection = None
        self.pca = None
        self.transform = None
        self.projection_speed = None
    

    def compute_projection(self):
        # Preprocess 
        rez = preprocess_pca(self.spike_times,self.spike_clusters,self.binsize,self.sigma,self.transform)
        self.raster = rez['raster']
        self.raster_smoothed = rez['raster_smoothed']
        self.raster_scaled = rez['raster_scaled']
        self.tbins = rez['tbins']
        self.cbins = rez['cbins']

        # Project
        self.projection,self.pca = project_pca(self.raster_scaled,self.tbins,self.ndims,self.t0,self.tf)

    def compute_projection_speed(self,ndims=3):
        self.projection_speed = compute_projection_speed(self.projection,ndims=ndims)


    def plot_by_speed(self,dims=[0,1,2],t0=None,tf=None,**kwargs):
        t0 = t0 or self.tbins[0]
        tf = tf or self.tbins[-1]
        if tf>self.spike_times.max():
            _log.warning(f'Requested max time {tf=} is greater than the last spike {self.spike_times.max():0.02f}s')
        s0,sf = np.searchsorted(self.tbins,[t0,tf])
        X_slice = self.projection[s0:sf,:]
        speed_slice = self.projection_speed[s0:sf]
        return(plot_projection(X_slice,dims,cvar=speed_slice,**kwargs))
    

    def plot_projection(self,dims=[0,1,2],t0=None,tf=None,cvar=None,**kwargs):
        t0 = t0 or self.tbins[0]
        tf = tf or self.tbins[-1]
        if tf>self.spike_times.max():
            _log.warning(f'Requested max time {tf=} is greater than the last spike {self.spike_times.max():0.02f}s')
        s0,sf = np.searchsorted(self.tbins,[t0,tf])
        X_slice = self.projection[s0:sf,:]
        if cvar is not None:
            cvar = cvar[s0:sf]
        return(plot_projection(X_slice,dims,cvar=cvar,**kwargs))
    
    def sync_var(self,x,x_t):
        return(remap_time_basis(x,x_t,self.tbins))
