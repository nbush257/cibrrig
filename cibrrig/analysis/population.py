from sklearn.decomposition import PCA
import numpy as np
from iblutil.numerical import bincount2D
from scipy.ndimage import gaussian_filter1d
import logging
from ..plot import plot_projection,plot_projection_line,plot_most_likely_dynamics
from ..utils.utils import validate_intervals, remap_time_basis
import pickle

logging.basicConfig()
_log = logging.getLogger("population")
_log.setLevel(logging.INFO)


def rasterize(spike_times, spike_clusters, binsize, FR=False):
    """
    Turn spike times and cluster ids into a 2D array of spikecounts per time bin
    Works on subsets of clusters (i.e., does not require the clusters to be continuous 0,1,2,3...)

    Args:
        binsize (float): Raster time bin size in seconds
    """
    raster, tbins, cbins = bincount2D(spike_times, spike_clusters, binsize)
    raster = raster.astype("float")
    # convert spike counts to spikes per second
    if FR:
        raster = raster / binsize
    return (raster, tbins, cbins)


def smooth_raster(raster, binsize, sigma):
    """
    Smooth a computed raster.
    Raster should be [cells x time]

    Args:
        binsize (float):
        sigma (_type_): _description_
    """
    sigma_scaled = sigma / binsize
    raster_smoothed = gaussian_filter1d(raster, sigma=sigma_scaled, axis=1)
    raster_smoothed[np.isnan(raster_smoothed)] = 0
    return raster_smoothed


def scale_raster(raster, transform="sqrt"):
    if transform == "sqrt":
        raster_scaled = np.sqrt(raster)
        raster_scaled[np.isinf(raster_scaled)] = 0
    else:
        raster_scaled = raster
    return raster_scaled


def preprocess_pca(spike_times, spike_clusters, binsize, sigma, transform="sqrt"):
    """
    wraps the preprocessing steps

    Args:
        binsize (_type_): _description_
        sigma (_type_): _description_
        transform (str, optional): _description_. Defaults to 'sqrt'.
    """
    raster, tbins, cbins = rasterize(spike_times, spike_clusters, binsize)
    raster_smoothed = smooth_raster(raster, binsize=binsize, sigma=sigma)
    raster_scaled = scale_raster(raster_smoothed, transform=transform)
    rez = dict(
        raster=raster,
        raster_smoothed=raster_smoothed,
        raster_scaled=raster_scaled,
        tbins=tbins,
        cbins=cbins,
    )
    return rez


def compute_projection_speed(X, ndims=3):
    """
    Compute the euclidean speed through PCA space
    :param X: PCA decompositions (2D numpy array: N_timepoints x N_dims)
    :param n: number of dimensions to use (int)
    :return: X_speed - 1D numpy array of PCA speed
    """

    if X.shape[0] <= X.shape[1]:
        raise Warning(
            f"Number of timepoints:{X.shape[0]} is fewer than number of dimensions:{X.shape[1]}. Confirm you do not need to transpose the matrix"
        )
    X_subset = X[:, :ndims]
    X_sd = np.diff(X_subset, axis=0)
    X_speed = np.concatenate([[0], np.sqrt(np.sum(X_sd**2, axis=1))])

    return X_speed


def compute_path_lengths(X, time_bins, t0, tf, ndims=3):
    """Compute the euclidean distance traveled between time t0 and tf
    in the projection space. Computes the distance along ndims

    Args:
        t0 (list or array): start time of the path
        tf (list or array): end time of the path
        ndims (int, optional): _description_. Defaults to 3.
    """
    assert isinstance(t0,(list,tuple,np.ndarray)), "t0 and tf need to be list or array, even if there is only one path to compute"
    
    path_lengths = np.zeros(len(t0)) * np.nan
    for ii, (start, stop) in enumerate(zip(t0, tf)):
        s0, sf = np.searchsorted(time_bins, [start, stop])
        diff = np.diff(X[s0:sf, :ndims], axis=0)
        squared_distances = np.sum(diff**2, axis=1)
        path_lengths[ii] = np.sum(np.sqrt(squared_distances))
    return path_lengths


def project_pca(raster, tbins, ndims=20, t0=None, tf=None):
    """
    Project the spiking data into the PCA space. Will fit on the interval between
    (t0,tf), and apply to the entire spiking data

    Raster should have shape [n_cells x time]. It will be transposed at fitting time

    If t0,tf is an array, fits on the intervals [t01,tf1],[t02,tf2]...
    If t0,tf is None, fits on all data
    """
    raster_train, tbins_train = _subset_raster(raster, tbins, t0, tf)
    pca = PCA(ndims).fit(raster_train.T)
    projected = pca.transform(raster.T)
    return (projected, pca)


def _subset_raster(raster, tbins, t0, tf):
    if t0 is None:
        t0 = np.array([np.min(tbins)])
    if tf is None:
        tf = np.array([np.max(tbins)])

    t0 = np.atleast_1d(t0)
    tf = np.atleast_1d(tf)
    raster_out = []
    tbins_out = []

    if len(t0) > 1:
        validate_intervals(t0, tf)

    for start, stop in zip(t0, tf):
        s0, sf = np.searchsorted(tbins, [start, stop])
        raster_out.append(raster[:, s0:sf])
        tbins_out.append(tbins[s0:sf])
    raster_out = np.hstack(raster_out)
    tbins_out = np.concatenate(tbins_out)

    return (raster_out, tbins_out)


def get_good_spikes(spikes, clusters):
    """
    Convinience function to return only good spikes
    """
    cluster_ids = clusters.metrics.query("bitwise_fail==0")["cluster_id"].values
    idx = np.isin(spikes.clusters, cluster_ids)
    for k in spikes.keys():
        spikes[k] = spikes[k][idx]
    _log.info(f'Total clusters: {clusters.metrics.shape[0]}\nQC pass {cluster_ids.size}\nKeeping only good clusters.')
    return (spikes, cluster_ids)

class Population:
    def __init__(
        self,
        spike_times,
        spike_clusters,
        ndims=None,
        binsize=0.005,
        sigma=0.01,
        t0=None,
        tf=None,
    ):
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
        self.has_ssm = False                

    def compute_projection(self):
        # Preprocess
        if self.projection is not None:
            _log.warning('Overwriting existing projection. May casue crashses')

        rez = preprocess_pca(
            self.spike_times,
            self.spike_clusters,
            self.binsize,
            self.sigma,
            self.transform,
        )
        self.raster = rez["raster"]
        self.raster_smoothed = rez["raster_smoothed"]
        self.raster_scaled = rez["raster_scaled"]
        self.tbins = rez["tbins"]
        self.cbins = rez["cbins"]

        # Project
        self.projection, self.pca = project_pca(
            self.raster_scaled, self.tbins, self.ndims, self.t0, self.tf
        )

    def compute_projection_speed(self, ndims=3):
        self.projection_speed = compute_projection_speed(self.projection, ndims=ndims)

    def plot_by_speed(self, dims=[0, 1, 2], t0=None, tf=None, **kwargs):
        t0 = t0 or self.tbins[0]
        tf = tf or self.tbins[-1]
        if tf > self.spike_times.max():
            _log.warning(
                f"Requested max time {tf=} is greater than the last spike {self.spike_times.max():0.02f}s"
            )
        s0, sf = np.searchsorted(self.tbins, [t0, tf])
        X_slice = self.projection[s0:sf, :]
        speed_slice = self.projection_speed[s0:sf]
        return plot_projection(X_slice, dims, cvar=speed_slice, **kwargs)

    def plot_projection(self, dims=[0, 1, 2], t0=None, tf=None, cvar=None, **kwargs):
        t0 = t0 or self.tbins[0]
        tf = tf or self.tbins[-1]
        if tf > self.spike_times.max():
            _log.warning(
                f"Requested max time {tf=} is greater than the last spike {self.spike_times.max():0.02f}s"
            )
        s0, sf = np.searchsorted(self.tbins, [t0, tf])
        if len(dims)>self.projection.shape[1]:
            _log.warning('Projection has fewer dims than requested. Only plotting first two requested')
            dims = dims[:2]

        X_slice = self.projection[s0:sf, :]
        if np.any(np.isnan(X_slice)):
            _log.error('Requested projection has NaNs')
            return
        if cvar is not None:
            cvar = cvar[s0:sf]
        return plot_projection(X_slice, dims, cvar=cvar, **kwargs)

    def plot_projection_line(self, dims=[0, 1, 2], t0=None, tf=None, cvar=None, **kwargs):
        t0 = t0 or self.tbins[0]
        tf = tf or self.tbins[-1]
        if tf > self.spike_times.max():
            _log.warning(
                f"Requested max time {tf=} is greater than the last spike {self.spike_times.max():0.02f}s"
            )
        if len(dims)>self.projection.shape[1]:
            _log.warning('Projection has fewer dims than requested. Only plotting first two requested')
            dims = dims[:2]

        s0, sf = np.searchsorted(self.tbins, [t0, tf])
        X_slice = self.projection[s0:sf, :]
        if np.any(np.isnan(X_slice)):
            _log.error('Requested projection has NaNs')
            return
        if cvar is not None:
            cvar = cvar[s0:sf]
        return plot_projection_line(X_slice, dims=dims, cvar=cvar, **kwargs)

    def sync_var(self, x, x_t):
        return remap_time_basis(x, x_t, self.tbins)

    def compute_path_lengths(self, t0, tf, ndims=3):
        return compute_path_lengths(self.projection, self.tbins, t0, tf, ndims)

    def compute_raster(self,binsize=0.005,t0=None,tf=None,FR=False,cluster_ids=None):
        '''
        Makes compute raster a method of the population
        De-couples the raster from the projection
        '''
        t0 = t0 or 0
        tf = tf or np.max(self.spike_times)
        idx = np.logical_and(
            self.spike_times>t0,
            self.spike_times<tf
        )
        if cluster_ids is not None:
            idx_clusters = np.isin(self.spike_clusters,cluster_ids)
            idx = np.logical_and(idx,idx_clusters)
        self.raster,self.tbins,self.cbins = rasterize(self.spike_times[idx],self.spike_clusters[idx],binsize,FR=FR)


    def load_rslds(self,ssm_fn,fit_on_load=False):
        '''
        Load a precomputed Linderman RSLDS model from a pickle file
        '''
        if self.projection is not None:
            _log.warning('Replacing PCA projection with RSLDS Latent')
        with open(ssm_fn,'rb') as fid:
            _log.debug(f'Loading ssm from {ssm_fn=}')
            dat = pickle.load(fid)
            _log.debug('\n\t'.join(['']))
        self.has_ssm = True
        # Unpack into population object

        self.pca=None
        self.sigma=None

        self.rslds = dat['rslds']
        self.q = dat['q']
        self.ndims = dat['D']
        self.binsize=dat['binsize']
        self.cbins = dat['cbins']

        # First check that the cluster ids match:
        # IF they do not match, determine if we can subset the spikes
        
        # There is a possibility that if you load the 
        # wrong data set and the incoming rslds model has more clusters than the 
        # given spiking data, that this will not complain and you will work with incompatible datasets. 
        # This is not likely I think if you are good about loading in datasets.
        existing_clusters = np.unique(self.spike_clusters)
        if not np.array_equal(existing_clusters,self.cbins):
            if np.setdiff1d(self.cbins,existing_clusters).size==0:
                # The existing clusters can all be found in the rslds dataset. We can subset
                _log.debug('All existing clusters were found. Creating raster')
                self.compute_raster(self.binsize,cluster_ids=self.cbins)
                self.raster = self.raster.astype('int') # For rslds
            else:
                _log.error('Existing clusters are not all found in the incoming RSLDS model. Did you load the right dataset')
        else:
            _log.debug('Incoming RSLDS matches existing population')

        #TODO: set bins where there is no projection equal to NaN
        idx = np.searchsorted(self.tbins,dat['tbins'])
        latent = dat['q'].mean_continuous_states[0]

        self.projection = np.full([self.tbins.shape[0],latent.shape[1]],np.nan)
        self.projection[idx,:] = latent

        if fit_on_load:
            # Must have a raster and that raster clusterids must match the fitted RSLDS
            self.smooth_rslds()

    def sim_rslds(self,duration):
        '''
        Simulate the latent for a given duration (in s)
        Adds a dictionary with keys ['discrete_states','continuous_states','emissions','sim_time'] to the population object 

        #TODO: pass initial state
        '''
        if not self.has_ssm:
            _log.error('SSM is not computed')
            self.sim_rslds = None
            return None

        # Determine time basis
        sim_time = np.arange(0,duration,self.binsize)
        nsamps = sim_time.shape[0]

        #Compute
        discrete_states,continuous_states,emissions = self.rslds.sample(nsamps)

        # Add as attribute
        self.sim_rslds = dict(
            discrete_states=discrete_states,
            sim_time=sim_time,
            continuous_states=continuous_states,
            emission=emissions
        )


    def transform_rslds(self,t0,tf):
        '''
        Compute the latent for a given interval [t0,tf]
        First checks if the latent already exists.
        return:
            elbos - estimate of fit
            posterior = slds posterior object
            latent - continuous states of the posterior

        '''
        if not self.has_ssm:
            _log.error('SSM is not computed')
            self.sim_rslds = None
            return None

        #TODO: place posterior data in projection
        sub_raster,sub_tbins = _subset_raster(self.raster,self.tbins,t0=t0,tf=tf)
        elbos,posterior = self.rslds.approximate_posterior(sub_raster.astype('int').T)
        latent = posterior.mean_continuous_states[0]

        idx = np.searchsorted(self.tbins,sub_tbins)
        self.projection[idx,:] = latent

        # Slot the data in to self.projection. Make sure tbins is correct
        return(elbos,posterior,latent)


    def plot_vectorfield(self,t0=None,tf=None, ax=None,nxpts=20,nypts=20,alpha=0.8,colors=None,**kwargs):
        '''
        Plot the flow field of the dynamics. 
        Wrapper to linderman Lab code in plot.py
        TODO: maybe include ability to plot 3D vector field?
        '''
        t0=t0 or 0
        tf=tf or self.tbins.max()
        s0,sf = np.searchsorted(self.tbins,[t0,tf])
        colors = colors or [f'C{x}' for x in range(7)]

        # Get plot ranges
        if ax is not None:
            xlim=ax.get_xlim()
            ylim=ax.get_ylim()
        else:
            mins = self.projection.min(0)
            maxs = self.projection.max(0)
            xlim = (mins[0],maxs[0])
            ylim = (mins[1],maxs[1])
        ax = plot_most_likely_dynamics(
            self.rslds,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            nxpts=nxpts,
            nypts=nypts,
            alpha=alpha,
            colors=colors,
            **kwargs

        )
        
        