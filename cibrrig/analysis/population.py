from sklearn.decomposition import PCA
import numpy as np
from iblutil.numerical import bincount2D
from scipy.ndimage import gaussian_filter1d
import logging
from cibrrig.plot import (
    plot_projection,
    plot_projection_line,
    plot_most_likely_dynamics,
    plot_projection_line_multicondition,
)
from cibrrig.utils.utils import validate_intervals, remap_time_basis
import pickle
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


logging.basicConfig()
_log = logging.getLogger("population")
_log.setLevel(logging.INFO)


def rasterize(spike_times, spike_clusters, binsize, FR=False):
    """
    Convert spike times and cluster IDs into a 2D array of spike counts per time bin.

    Args:
        spike_times (array-like): Array of spike times.
        spike_clusters (array-like): Array of cluster IDs corresponding to each spike time.
        binsize (float): Time bin size in seconds.
        FR (bool, optional): If True, convert spike counts to firing rates. Defaults to False.

    Returns:
        - raster (np.ndarray): 2D array of spike counts or firing rates [cells x time].
        - tbins (np.ndarray): Array of time bin edges.
        - cbins (np.ndarray): Array of unique cluster IDs.
    """
    raster, tbins, cbins = bincount2D(spike_times, spike_clusters, binsize)
    raster = raster.astype("float")
    # convert spike counts to spikes per second
    if FR:
        raster = raster / binsize
    return (raster, tbins, cbins)


def smooth_raster(raster, binsize, sigma):
    """
    Apply Gaussian smoothing to a spike raster.

    Args:
        raster (np.ndarray): 2D array of spike counts or firing rates [cells x time].
        binsize (float): Time bin size in seconds.
        sigma (float): Standard deviation of the Gaussian kernel in seconds.

    Returns:
        np.ndarray: Smoothed raster with the same shape as the input.
    """
    sigma_scaled = sigma / binsize
    raster_smoothed = gaussian_filter1d(raster, sigma=sigma_scaled, axis=1)
    raster_smoothed[np.isnan(raster_smoothed)] = 0
    return raster_smoothed


def scale_raster(raster, transform="sqrt"):
    """
    Apply a scaling transformation to the raster data.

    Args:
        raster (np.ndarray): 2D array of spike counts or firing rates [cells x time].
        transform (str, optional): Transformation to apply. Defaults to "sqrt".

    Returns:
        np.ndarray: Scaled raster with the same shape as the input.
    """

    if transform == "sqrt":
        raster_scaled = np.sqrt(raster)
        raster_scaled[np.isinf(raster_scaled)] = 0
    else:
        raster_scaled = raster
    return raster_scaled


def preprocess_pca(spike_times, spike_clusters, binsize, sigma, transform="sqrt"):
    """
    Preprocess spike data for PCA analysis.

    Args:
        spike_times (array-like): Array of spike times.
        spike_clusters (array-like): Array of cluster IDs corresponding to each spike time.
        binsize (float): Time bin size in seconds.
        sigma (float): Standard deviation for Gaussian smoothing in seconds.
        transform (str, optional): Transformation to apply to the smoothed raster. Defaults to "sqrt".

    Returns:
        dict: A dictionary containing preprocessed data:
            - raster: Original rasterized data
            - raster_smoothed: Smoothed raster
            - raster_scaled: Scaled and smoothed raster
            - tbins: Time bin edges
            - cbins: Unique cluster IDs
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
    assert isinstance(
        t0, (list, tuple, np.ndarray)
    ), "t0 and tf need to be list or array, even if there is only one path to compute"

    path_lengths = np.zeros(len(t0)) * np.nan
    for ii, (start, stop) in enumerate(zip(t0, tf)):
        s0, sf = np.searchsorted(time_bins, [start, stop])
        diff = np.diff(X[s0:sf, :ndims], axis=0)
        squared_distances = np.sum(diff**2, axis=1)
        path_lengths[ii] = np.sum(np.sqrt(squared_distances))
    return path_lengths


def project_pca(raster, tbins, ndims=20, t0=None, tf=None):
    """
    Project spiking data into PCA space. Will fit to the interval [t0,tf] and apply PCA to
    all times

    Args:
        raster (np.ndarray): 2D array of preprocessed spike data [cells x time].
        tbins (np.ndarray): Array of time bin edges.
        ndims (int, optional): Number of PCA dimensions to compute. Defaults to 20.
        t0 (float or array-like, optional): Start time(s) for fitting. Defaults to None (use all data).
        tf (float or array-like, optional): End time(s) for fitting. Defaults to None (use all data).

    Returns:
        - projected (np.ndarray): PCA projections of the input data [time x ndims].
        - pca (sklearn.decomposition.PCA): Fitted PCA object.
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
    _log.info(
        f"Total clusters: {clusters.metrics.shape[0]}\nQC pass {cluster_ids.size}\nKeeping only good clusters."
    )
    return (spikes, cluster_ids)


class Population:
    """
    A class for analyzing and visualizing population-level neural activity.

    Attributes:
        spike_times (array-like): Array of spike times.
        spike_clusters (array-like): Array of cluster IDs corresponding to each spike time.
        ndims (int): Number of dimensions for PCA projection.
        binsize (float): Time bin size in seconds.
        sigma (float): Standard deviation for Gaussian smoothing in seconds.
        t0 (float): Start time for analysis.
        tf (float): End time for analysis.
        raster (np.ndarray): Rasterized spike data.
        raster_smoothed (np.ndarray): Smoothed raster data.
        cbins (np.ndarray): Array of unique cluster IDs.
        tbins (np.ndarray): Array of time bin edges.
        projection (np.ndarray): PCA projection of the data.
        pca (sklearn.decomposition.PCA): Fitted PCA object.
        transform (str): Transformation applied to the raster.
        projection_speed (np.ndarray): Speed of movement through PCA space.
        has_ssm (bool): Whether a state-space model has been loaded.
    """

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
        """
        Initialize the Population object.

        Args:
            spike_times (array-like): Array of spike times.
            spike_clusters (array-like): Array of cluster IDs corresponding to each spike time.
            ndims (int, optional): Number of dimensions for PCA projection. Defaults to None.
            binsize (float, optional): Time bin size in seconds. Defaults to 0.005.
            sigma (float, optional): Standard deviation for Gaussian smoothing in seconds. Defaults to 0.01.
            t0 (float, optional): Start time for analysis. Defaults to None.
            tf (float, optional): End time for analysis. Defaults to None.
        """
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
        """
        Compute the PCA projection of the population activity.

        This method preprocesses the spike data, computes the raster,
        and projects it into PCA space.
        """
        # Preprocess
        if self.projection is not None:
            _log.warning("Overwriting existing projection. May casue crashses")

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
        """
        Compute the speed of movement through PCA space.

        Args:
            ndims (int, optional): Number of dimensions to use for speed calculation. Defaults to 3.
        """
        self.projection_speed = compute_projection_speed(self.projection, ndims=ndims)

    def plot_by_speed(self, dims=[0, 1, 2], t0=None, tf=None, **kwargs):
        """
        Plot the PCA projection colored by movement speed.

        Args:
            dims (list, optional): Dimensions to plot. Defaults to [0, 1, 2].
            t0 (float, optional): Start time for plotting. Defaults to None.
            tf (float, optional): End time for plotting. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            matplotlib.axes.Axes: The axes object containing the plot.
        """
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
        """
        Plot the low dimensional projection as a scatter
        Uses the known timebins of the projection to pass only a subset of the data to "plot_projection"

        Args:
            dims (list, optional): Which dimensions of the projection to plot. Can be 2 or 3 elements long. Defaults to [0, 1, 2].
            t0 (float, optional): start time of the plot in seconds. Defaults to None.
            tf (float_, optional): end time of the plot in seconds. Defaults to None.
            cvar (np.ndarray, optional): Variable to map to the color of the points. Defaults to None.

        Keyword Args passed to "plot_projection" from the plotting module.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
            color (str, optional): The color of the data points or lines. Default is "k" (black) if no color variable (`cvar`) is provided.
            alpha (float, optional): Transparency level for the points. Should be between 0 (fully transparent) and 1 (fully opaque).
                - Default is 0.5.
            lw (float, optional): Line width for plotting. Applies if plotting a line plot. Default is 0.5.
            vmin (float, optional): Minimum value for the colormap if a color variable (`cvar`) is used. If not specified, it's inferred from `cvar`.
            vmax (float, optional): Maximum value for the colormap if a color variable (`cvar`) is used. If not specified, it's inferred from `cvar`.
            colorbar_title (str, optional): Title for the colorbar. Default is an empty string.
            title (str, optional): Title for the plot. Only used in 3D plots. Default is an empty string.
            lims (list, optional): Axis limits for the plot. Should be a list of `[min, max]` values.
                - Default is [-4, 4] for 3D plots.
            pane_color (str, optional): Background color of the 3D panes in 3D plots. If `None`, default style is used.
            plot_colorbar (bool, optional): Whether to include a colorbar in the plot. Only applies to 3D plots when `cvar` is used. Default is True.
            s (float, optional): Size of the markers used in the scatter plot. Default is 1.
            cmap (str, optional): Colormap to use when plotting with a color variable (`cvar`). Default is "viridis".
            cvar (array-like, optional): An array of values used to color the data points. If provided, `cmap` is applied.
            s (float, optional): Size of the markers in the scatter plot. Default is 1.

        Returns:
            matplotlib.pyplot.axes: axes
        """
        t0 = t0 or self.tbins[0]
        tf = tf or self.tbins[-1]
        if tf > self.spike_times.max():
            _log.warning(
                f"Requested max time {tf=} is greater than the last spike {self.spike_times.max():0.02f}s"
            )
        s0, sf = np.searchsorted(self.tbins, [t0, tf])
        if len(dims) > self.projection.shape[1]:
            _log.warning(
                "Projection has fewer dims than requested. Only plotting first two requested"
            )
            dims = dims[:2]

        X_slice = self.projection[s0:sf, :]
        if np.any(np.isnan(X_slice)):
            _log.error("Requested projection has NaNs")
            return
        if cvar is not None:
            cvar = cvar[s0:sf]
        return plot_projection(X_slice, dims, cvar=cvar, **kwargs)

    def plot_projection_line(
        self, dims=[0, 1, 2], t0=None, tf=None, cvar=None, ax=None, **kwargs
    ):
        """Plot the low dimensional projection as a line
        Uses the known timebins of the projection to pass only a subset of the data to "plot_projection_line"
        Args:
            dims (list, optional): _description_. Defaults to [0, 1, 2].
            t0 (float, optional): _description_. Defaults to None.
            tf (float, optional): _description_. Defaults to None.
            cvar (np.ndarray, optional): _description_. Defaults to None.
            ax (matplotlib.pyplot.axes, optional): _description_. Defaults to None.

        Keyword Args:
            intervals (np.ndarray,optional): Start and end times for each condition.
            color (str, optional): Color of the line if cvar is not provided. Default is "k" (black).
            stim_color (str, optional): Color of the line during intervals (if provided)
            alpha (float, optional): The alpha blending value, between 0 (transparent) and 1 (opaque). Default is 0.5.
            lw (float, optional): The line width. Default is 0.5.
            vmin (float, optional): Minimum of the colormap range. If not provided, it's inferred from cvar.
            vmax (float, optional): Maximum of the colormap range. If not provided, it's inferred from cvar.
            colorbar_title (str, optional): Title for the colorbar. Default is an empty string.
            title (str, optional): Title for the plot. Only used in 3D plots. Default is an empty string.
            lims (list, optional): The x, y, (and z for 3D) limits of the plot as [min, max]. Only used in 3D plots. Default is [-4, 4].
            pane_color (color, optional): Color of the panes in 3D plots. If None, default matplotlib style is used.
            plot_colorbar (bool, optional): Whether to plot the colorbar. Only used in 3D plots. Default is True.

        Returns:
            Axes: The axes object containing the plot.
        """
        t0 = t0 or self.tbins[0]
        tf = tf or self.tbins[-1]
        if tf > self.spike_times.max():
            _log.warning(
                f"Requested max time {tf=} is greater than the last spike {self.spike_times.max():0.02f}s"
            )
        if len(dims) > self.projection.shape[1]:
            _log.warning(
                "Projection has fewer dims than requested. Only plotting first two requested"
            )
            dims = dims[:2]
        if len(dims) == 2:
            kwargs.pop("lims", None)

        s0, sf = np.searchsorted(self.tbins, [t0, tf])
        X_slice = self.projection[s0:sf, :]
        intervals = kwargs.pop("intervals", None)
        skip_control=False
        if 'stim_color' in kwargs:
            stim_color = kwargs.pop("stim_color")
        else:
            stim_color = "C1"
        if 'base_color' in kwargs:
            base_color = kwargs.pop("base_color")
            if base_color == "none":
                skip_control = True
            else:
                skip_control = False
        else:
            base_color = "C0"
        
        if len(dims) == 3:
            lims = kwargs.get("lims", None)
            if lims is None:
                lim = np.nanmax(np.abs(X_slice[:, dims]))
                lims = [-lim, lim]
                kwargs["lims"] = lims
                
        if intervals is not None:
            if not skip_control:
                intervals_baseline = []
                _temp = t0
                for _t0, _tf in intervals:
                    intervals_baseline.append([_temp, _t0])
                    _temp = _tf
                if tf > _temp:
                    intervals_baseline.append([_temp, tf])
                intervals_baseline = np.array(intervals_baseline)

                base_colors = [base_color for _ in range(intervals_baseline.shape[0])]
                ax = plot_projection_line_multicondition(
                    X_slice,
                    self.tbins[s0:sf],
                    intervals_baseline,
                    colors=base_colors,
                    dims=dims,
                    ax=ax,
                    **kwargs,
                )
                
            stim_colors = [stim_color for _ in range(intervals.shape[0])]
            ax = plot_projection_line_multicondition(
                self.projection,
                self.tbins,
                intervals,
                colors=stim_colors,
                dims=dims,
                ax=ax,
                **kwargs,
            )
            ax.legend(["Control", "Stim"])
            legend_elements = [
                Line2D([0], [0], color=stim_color, lw=2, label="Stim"),  # Cyan line
                Line2D([0], [0], color=base_color, lw=2, label="Control"),  # Black line
            ]
            ax.legend(
                handles=legend_elements, loc="upper left", bbox_to_anchor=(0.9, 0.9)
            )

            return ax

        if np.any(np.isnan(X_slice)):
            _log.error("Requested projection has NaNs")
            return
        if cvar is not None:
            cvar = cvar[s0:sf]
        return plot_projection_line(X_slice, dims=dims, cvar=cvar, ax=ax, **kwargs)

    def sync_var(self, x, x_t):
        """Resample an exogenous signal with the projection

        Args:
            x (np.ndarray): exogenous signal to resample
            x_t (np.ndarray): sample times for x in seconds

        Returns:
            np.ndarray: y - resampled signal
        """
        return remap_time_basis(x, x_t, self.tbins)

    def compute_path_lengths(self, t0, tf, ndims=3):
        """Compute the euclidean distance traveled between time t0 and tf
        in the projection space. Computes the distance along ndims

        Args:
            t0 (list or array): start time of the path
            tf (list or array): end time of the path
            ndims (int, optional): number of dimensions to include in computation. Defaults to 3.
        """
        return compute_path_lengths(self.projection, self.tbins, t0, tf, ndims)

    def compute_raster(
        self, binsize=0.005, t0=None, tf=None, FR=False, cluster_ids=None
    ):
        """
        Convert spike times and cluster IDs into a 2D array of spike counts per time bin.

        Args:
            t0 (float): start time of raster in seconds
            tf (float): stop time of raster in seconds
            binsize (float): Time bin size in seconds.
            FR (bool, optional): If True, convert spike counts to firing rates. Defaults to False.
        """
        t0 = t0 or 0
        tf = tf or np.max(self.spike_times)
        idx = np.logical_and(self.spike_times > t0, self.spike_times < tf)
        if cluster_ids is not None:
            idx_clusters = np.isin(self.spike_clusters, cluster_ids)
            idx = np.logical_and(idx, idx_clusters)
        self.raster, self.tbins, self.cbins = rasterize(
            self.spike_times[idx], self.spike_clusters[idx], binsize, FR=FR
        )

    def load_rslds(self, ssm_fn, fit_on_load=False):
        """Load a precomputed Linderman RSLDS model from a pickle file


        Args:
            ssm_fn (Path): filename of the computed rslds model
            fit_on_load (bool, optional): Not yet implemented. Defaults to False.
        """

        if self.projection is not None:
            _log.warning("Replacing PCA projection with RSLDS Latent")
        with open(ssm_fn, "rb") as fid:
            _log.debug(f"Loading ssm from {ssm_fn=}")
            dat = pickle.load(fid)
            _log.debug("\n\t".join([""]))
        self.has_ssm = True
        # Unpack into population object

        self.pca = None
        self.sigma = None
        if self.binsize != dat["binsize"]:
            _log.warning("Recomputing time bins")
            _t0 = self.spike_times.min()
            _tf = self.spike_times.max()
            self.tbins = np.arange(_t0, _tf, dat["binsize"])

        self.rslds = dat["rslds"]
        self.q = dat["q"]
        self.ndims = dat["D"]
        self.binsize = dat["binsize"]
        self.cbins = dat["cbins"]

        # First check that the cluster ids match:
        # IF they do not match, determine if we can subset the spikes

        # There is a possibility that if you load the
        # wrong data set and the incoming rslds model has more clusters than the
        # given spiking data, that this will not complain and you will work with incompatible datasets.
        # This is not likely I think if you are good about loading in datasets.
        existing_clusters = np.unique(self.spike_clusters)
        if not np.array_equal(existing_clusters, self.cbins):
            if np.setdiff1d(self.cbins, existing_clusters).size == 0:
                # The existing clusters can all be found in the rslds dataset. We can subset
                _log.debug("All existing clusters were found. Creating raster")
                self.compute_raster(self.binsize, cluster_ids=self.cbins)
                self.raster = self.raster.astype("int")  # For rslds
            else:
                _log.error(
                    "Existing clusters are not all found in the incoming RSLDS model. Did you load the right dataset"
                )
        else:
            _log.debug("Incoming RSLDS matches existing population")

        # TODO: set bins where there is no projection equal to NaN
        idx = np.searchsorted(self.tbins, dat["tbins"])
        latent = dat["q"].mean_continuous_states[0]

        self.projection = np.full([self.tbins.shape[0], latent.shape[1]], np.nan)
        self.projection[idx, :] = latent

        if fit_on_load:
            # Must have a raster and that raster clusterids must match the fitted RSLDS
            raise NotImplementedError("fit on load no tet implemented")
            self.smooth_rslds()

    def sim_rslds(self, duration):
        """Simulate the latent for a given duration (in s)
        Adds a dictionary with keys ['discrete_states','continuous_states','emissions','sim_time'] to the population object

        Args:
            duration (float): Time duration to simulate in seconds

        #TODO: pass initial state
        """
        if not self.has_ssm:
            _log.error("SSM is not computed")
            self.sim_rslds = None
            return None

        # Determine time basis
        sim_time = np.arange(0, duration, self.binsize)
        nsamps = sim_time.shape[0]

        # Compute
        discrete_states, continuous_states, emissions = self.rslds.sample(nsamps)

        # Add as attribute
        self.sim_rslds = dict(
            discrete_states=discrete_states,
            sim_time=sim_time,
            continuous_states=continuous_states,
            emission=emissions,
        )

    def transform_rslds(self, t0, tf):
        """Compute the latent for a given interval [t0,tf]
        First checks if the latent already exists.

        Args:
            t0 (float): Start time in seconds
            tf (float): End time in seconds

        Returns:
            (np.ndarray): ELBOS estimate of fit quality
            (slds.posterior): slds posterior object
            (np.ndarray): latent -- continuous states of the posterior
        """

        if not self.has_ssm:
            _log.error("SSM is not computed")
            self.sim_rslds = None
            return None

        sub_raster, sub_tbins = _subset_raster(self.raster, self.tbins, t0=t0, tf=tf)
        elbos, posterior = self.rslds.approximate_posterior(sub_raster.astype("int").T)
        latent = posterior.mean_continuous_states[0]

        idx = np.searchsorted(self.tbins, sub_tbins)
        self.projection[idx, :] = latent

        # Slot the data in to self.projection. Make sure tbins is correct
        return (elbos, posterior, latent)

    def plot_vectorfield(
        self,
        t0=None,
        tf=None,
        ax=None,
        nxpts=20,
        nypts=20,
        alpha=0.8,
        colors=None,
        xlim=None,
        ylim=None,
        zval=None,
        **kwargs,
    ):
        """
        Plot the flow field of the dynamics.
        Wrapper to linderman Lab code in plot.py
        TODO: maybe include ability to plot 3D vector field?
        """
        t0 = t0 or 0
        tf = tf or self.tbins.max()
        s0, sf = np.searchsorted(self.tbins, [t0, tf])
        colors = colors or [f"C{x}" for x in range(7)]

        # Get plot ranges
        if ax is not None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        elif xlim is None:
            mins = np.nanmin(self.projection, 0)
            maxs = np.nanmax(self.projection, 0)
            xlim = (mins[0], maxs[0])
            ylim = (mins[1], maxs[1])
        ax = plot_most_likely_dynamics(
            self.rslds,
            xlim=xlim,
            ylim=ylim,
            ax=ax,
            nxpts=nxpts,
            nypts=nypts,
            alpha=alpha,
            colors=colors,
            zval=None,
            **kwargs,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        return ax
