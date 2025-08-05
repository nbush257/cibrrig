import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from one.alf.io import AlfBunch
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator

try:
    from brainbox.plot import driftmap
    from brainbox.processing import bincount2D
    import brainbox.singlecell as bbsc

    has_brainbox = True
except ImportError:
    print("No brainbox package")
    has_brainbox = False

from cibrrig.utils.utils import parse_opto_log, validate_intervals, weighted_histogram

# Maps laser wavelengths to hex codes
laser_colors = {473: "#00b7ff", 565: "#d2ff00", 635: "#ff0000"}


def plot_laser(laser_in, **kwargs):
    """
    Flexibly overlay laser stimulation.

    Args:
        laser_in (AlfBunch or array-like): The laser data to be plotted. Can be an AlfBunch object or an array of intervals.
        **kwargs: Additional keyword arguments to be passed to the plotting functions. These may include:
            mode (str): The plotting mode. Options are "shade", "bar", "vline", or any other (defaults to steps).
            ax (matplotlib.axes.Axes): The axes object to plot on. If None, a new figure and axes will be created.
            amp_label (str): Label for the amplitude axis when plotting amplitudes.
            wavelength (int): Laser wavelength in nm, used to determine the color of the plot.
            alpha (float or array-like): The alpha (transparency) value(s) for shaded areas.
            color (str or tuple): The color to use for plotting. If not provided, a default color based on wavelength is used.
            query (str): Query string to filter the data (only used in _plot_laser_log).
            rotation (int): Rotation angle for text annotations (only used in _plot_laser_log).
            fontsize (int): Font size for text annotations (only used in _plot_laser_log).
            Any other keyword arguments accepted by matplotlib plotting functions.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.

    Notes:
        This function determines the appropriate plotting method based on the input type:
        - If laser_in is an AlfBunch object with a 'category' key, it calls _plot_laser_log.
        - If laser_in is an AlfBunch object without a 'category' key, it calls _plot_laser_alf.
        - For other input types, it calls _plot_laser_intervals.

        The specific kwargs used may vary depending on which underlying plotting function is called.
    """

    if isinstance(laser_in, AlfBunch):
        if "category" in laser_in.keys():
            ax = _plot_laser_log(laser_in, **kwargs)
        else:
            ax = _plot_laser_alf(laser_in, **kwargs)
    else:
        ax = _plot_laser_intervals(laser_in, **kwargs)
    return ax


def _plot_laser_alf(laser_in, **kwargs):
    """
    Plot laser data from a "laser" AlfBunch.

    Args:
        laser_in (AlfBunch): The AlfBunch object containing laser data.
        **kwargs: Additional keyword arguments to be passed to _plot_laser_intervals.

    Notes:
        This function extracts intervals and amplitudes from the AlfBunch object and calls
        _plot_laser_intervals with the appropriate parameters. It determines whether to use
        milliwatts or volts for the amplitude label based on the available keys in laser_in.
    """
    intervals = laser_in.intervals
    if "amplitudesMilliwatts" in laser_in.keys():
        amplitudes = laser_in.amplitudesMilliwatts
        amp_label = "mW"
    elif "amplitudesVolts" in laser_in.keys():
        amplitudes = laser_in.amplitudesVolts
        amp_label = "command volts"
    else:
        amplitudes = None

    _plot_laser_intervals(intervals, amplitudes, amp_label=amp_label, **kwargs)


def _plot_laser_intervals(
    intervals,
    amplitudes=None,
    ax=None,
    mode="shade",
    amp_label="",
    wavelength=473,
    alpha=0.2,
    **kwargs,
):
    """
    Plot laser intervals from arrays.

    Args:
        intervals (array-like): Array of laser intervals, where each interval is [start_time, end_time].
        amplitudes (array-like, optional): Array of amplitude values corresponding to each interval.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes will be created.
        mode (str, optional): The plotting mode. Options are "shade", "bar", "vline", or any other (defaults to steps).
        amp_label (str, optional): Label for the amplitude axis when plotting amplitudes.
        wavelength (int, optional): Laser wavelength in nm, used to determine the color of the plot. Default is 473.
        alpha (float or array-like, optional): The alpha (transparency) value(s) for shaded areas. Default is 0.2.
        **kwargs: Additional keyword arguments to be passed to the plotting functions.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.

    Notes:
        This function supports multiple plotting modes:
        - "shade": Shades the intervals on the plot.
        - "bar": Plots horizontal bars for each interval.
        - "vline": Plots vertical lines at the start of each interval.
        - Any other mode defaults to plotting steps of the amplitudes.

        The function handles color selection based on the wavelength and can use a list
        of alpha values for varying transparency across intervals.
    """
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)

    try:
        iter(alpha)
        alpha_list = True
    except Exception:
        alpha_list = False

    if kwargs.get("color") is not None:
        color = kwargs.pop("color")
    else:
        color = laser_colors[wavelength]

    if mode == "shade":
        for ii, stim in enumerate(intervals):
            aa = alpha[ii] if alpha_list else alpha
            ax.axvspan(stim[0], stim[1], color=color, alpha=aa, **kwargs)
    elif mode == "bar":
        yy = ax.get_ylim()[1]
        yy = np.ones_like(intervals[:, 0]) * yy * 0.95
        ax.hlines(
            yy,
            intervals[:, 0],
            intervals[:, 1],
            color=color,
            **kwargs,
        )
    elif mode == "vline":
        y0 = ax.get_ylim()[0]
        y0 = np.ones_like(intervals[:, 0]) * y0

        y1 = ax.get_ylim()[1]
        y1 = np.ones_like(intervals[:, 0]) * y1
        ax.vlines(intervals[:, 0], y0, y1, color=color, **kwargs)
    else:
        # interleave zeros for the offsets
        print(f"mode {mode} not found. Plotting as steps")
        ax = ax.twinx()
        if amplitudes is None:
            new_amps = np.vstack(
                [np.zeros_like(intervals[:, 0]), np.ones_like(intervals[:, 0])]
            ).T.ravel()
        else:
            new_amps = np.vstack([np.zeros_like(amplitudes), amplitudes]).T.ravel()
        ax.step(intervals.ravel(), new_amps, color=laser_colors[wavelength], **kwargs)
        ax.set_ylabel(amp_label)
    plt.xlabel("Time (s)")
    return ax


def _plot_laser_log(log, query=None, rotation=45, fontsize=6, **kwargs):
    """
    Plot laser data from a "log" AlfBunch object.

    Args:
        log (AlfBunch): The AlfBunch object containing laser log data.
        query (str, optional): Query string to filter the data. Default is None.
        rotation (int, optional): Rotation angle for text annotations. Default is 45.
        fontsize (int, optional): Font size for text annotations. Default is 6.
        **kwargs: Additional keyword arguments to be passed to _plot_laser_intervals.

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.

    Notes:
        This function extracts opto data from the log, plots the intervals using _plot_laser_intervals,
        and adds text annotations for each interval. It handles both milliwatt and voltage amplitudes.
    """
    opto_df = log.to_df().query('category=="opto"')
    intervals = opto_df[["start_time", "end_time"]].values
    if "amplitude_mw" in opto_df.keys():
        amps = opto_df["amplitude_mw"]
        amp_units = "mW"
    else:
        amps = opto_df["amplitude"]
        amp_units = "command_volts"

    ax = _plot_laser_intervals(
        intervals, amplitudes=amps, amp_label=amp_units, **kwargs
    )
    if query:
        opto_df = opto_df.query(query)
    for _, rr in opto_df.iterrows():
        s = parse_opto_log(rr)
        # TODO: Fix text going big
        ax.text(
            np.mean([rr.start_time, rr.end_time]),
            plt.gca().get_ylim()[1],
            s,
            rotation=rotation,
            fontsize=fontsize,
        )
    return ax


def _create_ax(dims, projection=None):
    """
    Create a new figure and axis.

    Args:
        dims (list): Dimensions to plot.
        projection (str, optional): Type of projection for 3D plots.

    Returns:
        tuple: Figure and Axes objects.
    """
    f = plt.figure()
    ax = f.add_subplot(projection=projection)
    return f, ax


def _setup_colorbar(ax, p, vmin, vmax, colorbar_title):
    """
    Set up a colorbar for the plot.

    Args:
        ax (Axes): The axes object to add the colorbar to.
        p: The plot object to create the colorbar from.
        vmin (float): Minimum value for the colorbar.
        vmax (float): Maximum value for the colorbar.
        colorbar_title (str): Title for the colorbar.
    """
    cbar = plt.colorbar(p, ax=ax, pad=0.1, orientation="horizontal", location="top")
    cbar.set_label(colorbar_title)
    if vmin >= 0:
        cbar.set_ticks([vmin, vmax])
    else:
        # Check if vmin and vmax are  close to -pi and pi
        if np.isclose(vmin, -np.pi,rtol=0.1) and np.isclose(vmax, np.pi, rtol=0.1):
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels([r"$\pi$", "0", r"$\pi$"])
        else:
            cbar.set_ticks([vmin, 0, vmax])
    cbar.outline.set_edgecolor("none")

    cbar.solids.set_alpha(1)


def plot_projection_line_multicondition(
    X, tbins, intervals, colors, dims=[0, 1], ax=None, alpha=0.5, lw=1, **kwargs
):
    """
    Plot low-D projection with unique coloring for the given intervals.

    Args:
        X (array): Data to plot.
        tbins (array): Time bins.
        intervals (array): Start and end times for each condition.
        colors (list): Colors for each condition.
        dims (list): Dimensions to plot.
        ax (Axes, optional): Axes to plot on.
        alpha (float): Alpha value for transparency.
        lw (float): Line width.

    Returns:
        Axes: The axes object containing the plot.
    """
    validate_intervals(intervals[:, 0], intervals[:, 1], overlap_ok=True)
    assert len(colors) == intervals.shape[0]

    if ax is None:
        _, ax = _create_ax(dims, projection="3d" if len(dims) == 3 else None)

    for (t0, tf), cc in zip(intervals, colors):
        s0, sf = np.searchsorted(tbins, [t0, tf])
        X_sub = X[s0 - 1 : sf, :]
        ax = plot_projection_line(
            X_sub, dims=dims, cvar=None, color=cc, alpha=alpha, ax=ax, lw=lw, **kwargs
        )
    return ax


def plot_projection_line(X, cvar=None, dims=[0, 1], cmap="viridis", **kwargs):
    """
    Plot low-d projection as a line. Optionally

    Args:
        X (array): Data to plot.
        cvar (array, optional): Color variable.
        dims (list): Dimensions to plot.
        cmap (str): Colormap to use.
        **kwargs: Additional keyword arguments.

    Keyword Args:
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, a new figure and axes will be created.
        color (str, optional): Color of the line if cvar is not provided. Default is "k" (black).
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
    if len(dims) == 2:
        return _plot_projection_line_2D(X, cvar, dims, cmap=cmap, **kwargs)
    elif len(dims) == 3:
        ax = _plot_projection_line_3D(X, cvar, dims=dims, cmap=cmap, **kwargs)
    else:
        raise ValueError("Number of dims must be two or three")
    return ax


def _plot_projection_line_2D(
    X,
    cvar=None,
    dims=[0, 1],
    cmap="viridis",
    color="k",
    ax=None,
    alpha=0.5,
    vmin=None,
    vmax=None,
    lw=0.5,
    colorbar_title="",
    plot_colorbar=True,
    **kwargs,
):
    """
    Plot 2D projection line.

    Args:
        X (array): Data to plot.
        cvar (array, optional): Color variable.
        dims (list): Dimensions to plot.
        cmap (str): Colormap to use.
        color (str): Color for the line if cvar is None.
        ax (Axes, optional): Axes to plot on.
        alpha (float): Alpha value for transparency.
        vmin (float, optional): Minimum value for colormap.
        vmax (float, optional): Maximum value for colormap.
        lw (float): Line width.
        colorbar_title (str): Title for the colorbar.
        **kwargs: Additional keyword arguments.

    Returns:
        Axes: The axes object containing the plot.
    """
    if ax is None:
        _, ax = _create_ax(dims)

    segments = np.stack([X[:-1, dims], X[1:, dims]], axis=1)

    use_arrow = kwargs.pop("use_arrow", None)
    mutation_scale = kwargs.pop("mutation_scale", 10)
    # TODO: implement an arrow in the middle of the line
    multi_arrow = kwargs.pop("multi_arrow", False)

    if use_arrow:
        if segments.shape[0] == 0:
            use_arrow = False
        elif segments.shape[0] < 2:
            multi_arrow = False
        else:
            (
                a,
                b,
            ) = segments[-1]
            arrow = FancyArrowPatch(
                a, b, arrowstyle="-|>", color=color, lw=lw, alpha=alpha, mutation_scale=mutation_scale
            )
            segments = segments[:-1]
        
        if multi_arrow:
            
            (a,b) = segments[1]
            arrow2 = FancyArrowPatch(
                a, b, arrowstyle="-|>", color=color, lw=lw, alpha=alpha, mutation_scale=mutation_scale
            )

    _ = kwargs.pop("s", None)
    lc = LineCollection(segments, alpha=alpha, lw=lw, **kwargs)
    if cvar is not None:
        if vmin is None:
            vmin = np.min(cvar)
        if vmax is None:
            vmax = np.max(cvar)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        lc.set_array(cvar)
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        if plot_colorbar:
            _setup_colorbar(ax, lc, vmin, vmax, colorbar_title)
    else:
        lc.set_color(color)

    ax.add_collection(lc)
    if use_arrow:
        ax.add_patch(arrow)
        if multi_arrow:
            ax.add_patch(arrow2)

    sns.despine()
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dim {dims[0]+1}")
    ax.set_ylabel(f"Dim {dims[1]+1}")

    # if cvar is not None and plot_colorbar:
    #     cbar = plt.colorbar(
    #         lc, ax=ax, pad=0.1, orientation="horizontal", location="top"
    #     )
    #     cbar.set_label(colorbar_title)
    #     cbar.set_ticks([vmin, 0, vmax])
    #     cbar.solids.set_alpha(1)

    return ax


def _plot_projection_line_3D(
    X,
    cvar=None,
    dims=[0, 1, 2],
    cmap="viridis",
    color="k",
    ax=None,
    title="",
    alpha=0.5,
    lims=None,
    pane_color=None,
    colorbar_title="",
    plot_colorbar=True,
    vmin=None,
    vmax=None,
    lw=0.5,
    **kwargs,
):
    """
    Plot 3D projection line.

    Args:
        X (array): Data to plot.
        cvar (array, optional): Color variable.
        dims (list): Dimensions to plot.
        cmap (str): Colormap to use.
        color (str): Color for the line if cvar is None.
        ax (Axes3D, optional): 3D axes to plot on.
        title (str): Title for the plot.
        alpha (float): Alpha value for transparency.
        lims (list): Limits for the axes.
        pane_color: Color for the panes.
        colorbar_title (str): Title for the colorbar.
        plot_colorbar (bool): Whether to plot the colorbar.
        vmin (float, optional): Minimum value for colormap.
        vmax (float, optional): Maximum value for colormap.
        lw (float): Line width.
        **kwargs: Additional keyword arguments.

    Returns:
        Axes3D: The 3D axes object containing the plot.
    """
    if ax is None:
        _, ax = _create_ax(dims, projection="3d")

    segments = np.stack([X[:-1, dims], X[1:, dims]], axis=1)
    lc = Line3DCollection(segments, alpha=alpha, lw=lw, **kwargs)

    if cvar is not None:
        vmin = vmin or np.min(cvar)
        vmax = vmax or np.max(cvar)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors = plt.get_cmap(cmap)(norm(cvar[:-1]))
        lc.set_color(colors)
        if plot_colorbar:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array(colors)
            _setup_colorbar(ax, sm, vmin, vmax, colorbar_title)
    else:
        lc.set_color(color)

    ax.add_collection(lc)
    ax.autoscale()
    if lims is None:
        lim = np.nanmax(np.abs(X[:, dims]))
        lims = [-lim, lim]

    _clean_3d_axes(ax, title, dims, pane_color, lims=lims)

    return ax


def plot_projection(X, dims, **kwargs):
    """
    Plot projection in 2D or 3D.

    Args:
        X (array): Data to plot.
        dims (list): Dimensions to plot.
        **kwargs: Additional keyword arguments.

    Keyword Args:
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
        tuple: Figure and Axes objects.
    """
    if len(dims) == 2:
        return plot_2D_projection(X, dims, **kwargs)
    elif len(dims) == 3:
        return plot_3D_projection(X, dims, **kwargs)
    else:
        raise ValueError(f"Number of plotted dimensions must be 2 or 3. {dims=}")


def plot_3D_projection(
    X,
    dims=[0, 1, 2],
    cvar=None,
    ax=None,
    title="",
    s=1,
    vmin=None,
    vmax=None,
    cmap="viridis",
    c="k",
    alpha=0.2,
    lims=None,
    plot_colorbar=True,
    colorbar_title="",
    pane_color=None,
    **kwargs,
):
    """
    Plot 3D projection.

    Args:
        X (array): Data to plot.
        dims (list): Dimensions to plot.
        cvar (array, optional): Color variable.
        ax (Axes3D, optional): 3D axes to plot on.
        title (str): Title for the plot.
        s (float): Size of the markers.
        vmin (float, optional): Minimum value for colormap.
        vmax (float, optional): Maximum value for colormap.
        cmap (str): Colormap to use.
        c (str): Color for the markers if cvar is None.
        alpha (float): Alpha value for transparency.
        lims (list): Limits for the axes.
        plot_colorbar (bool): Whether to plot the colorbar.
        colorbar_title (str): Title for the colorbar.
        pane_color: Color for the panes.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: Figure and Axes3D objects.
    """
    assert len(dims) == 3, f"Must choose 3 dimensions to plot. Chose {dims}"

    if ax is None:
        _, ax = _create_ax(dims, projection="3d")
    if cvar is None:
        p = ax.scatter(
            X[:, dims[0]],
            X[:, dims[1]],
            X[:, dims[2]],
            c=cvar,
            s=s,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
    else:
        p = ax.scatter(
            X[:, dims[0]],
            X[:, dims[1]],
            X[:, dims[2]],
            c=c,
            s=s,
            alpha=alpha,
            cmap=None,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
    if cvar is not None and plot_colorbar:
        _setup_colorbar(
            ax, p, vmin or np.min(cvar), vmax or np.max(cvar), colorbar_title
        )

    ax.autoscale()
    if lims is None:
        lim = np.nanmax(np.abs(X[:, dims]))
        lims = [-lim, lim]
    _clean_3d_axes(ax, title, dims, pane_color, lims=lims)

    return ax


def plot_2D_projection(
    X,
    dims=[0, 1],
    cvar=None,
    ax=None,
    title="",
    s=1,
    vmin=None,
    vmax=None,
    cmap="viridis",
    c="C1",
    alpha=0.2,
    lims=[-4, 4],
    plot_colorbar=True,
    colorbar_title="",
):
    """
    Plot 2D projection.

    Args:
        X (array): Data to plot.
        dims (list): Dimensions to plot.
        cvar (array, optional): Color variable.
        ax (Axes, optional): Axes to plot on.
        title (str): Title for the plot.
        s (float): Size of the markers.
        vmin (float, optional): Minimum value for colormap.
        vmax (float, optional): Maximum value for colormap.
        cmap (str): Colormap to use.
        c (str): Color for the markers if cvar is None.
        alpha (float): Alpha value for transparency.
        lims (list): Limits for the axes.
        plot_colorbar (bool): Whether to plot the colorbar.
        colorbar_title (str): Title for the colorbar.

    Returns:
        tuple: Figure and Axes objects.
    """
    assert len(dims) == 2, f"Must choose 2 dimensions to plot. Chose {dims}"

    if ax is None:
        _, ax = _create_ax(dims)

    if cvar is None:
        p = ax.scatter(
            X[:, dims[0]],
            X[:, dims[1]],
            c=c,
            s=s,
            alpha=alpha,
            cmap=None,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        p = ax.scatter(
            X[:, dims[0]],
            X[:, dims[1]],
            c=cvar,
            s=s,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    if cvar is not None and plot_colorbar:
        _setup_colorbar(
            ax, p, vmin or np.min(cvar), vmax or np.max(cvar), colorbar_title
        )

    ax.set_title(title)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel(f"Dim {dims[0]+1}")
    ax.set_ylabel(f"Dim {dims[1]+1}")
    ax.spines[["right", "top"]].set_visible(False)

    return ax.get_figure(), ax


def plot_polar_average(
    x,
    y,
    t,
    ax=None,
    t0=None,
    tf=None,
    color="k",
    bins=50,
    multi="sem",
    alpha=0.3,
    **plot_kwargs,
):
    """
    Plot covariate `y` as a function of phase `x` on a polar.

    If t0,tf are arrays, will average over multiple intervals

    Args:
        x (1D numpy array): Phase data with values in the range [-pi, pi].
        y (1D numpy array): Signal data to be plotted against `x`.
        t (1D numpy array): Time data corresponding to `x` and `y`.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
        t0 (int, float, or list, optional): Start time(s) for epoch selection. If a list or array, averages over multiple epochs are computed. Defaults to None.
        tf (int, float, or list, optional): End time(s) for epoch selection, matching the format of `t0`. Defaults to None.
        color (str or list, optional): Line color(s) for the plot. Defaults to 'k'.
        bins (int, optional): Number of bins for the polar histogram. Defaults to 50.
        multi (str, optional): Specifies the method for calculating the shaded region. Options are 'std' for standard deviation or 'sem' for standard error of the mean. Defaults to 'sem'.
        alpha (float, optional): Transparency of the shaded region. Defaults to 0.3.
        **plot_kwargs: Additional keyword arguments passed to `ax.plot`.

    Returns:
        tuple: A tuple containing:
            - `f`: The created figure object (or None if `ax` was provided).
            - `ax`: The axes object used for plotting.
            - `y_polar_out`: 2D numpy array of the polar data averaged over epochs.
            - `phase_bins`: Phase bin centers for the plot.

    Example:
        >>> plot_polar_average(x, y, t, t0=0, tf=10, color='b', bins=30, multi='std')
    """

    try:
        iter(t0)
    except Exception:
        t0 = [t0]

    try:
        iter(tf)
    except Exception:
        tf = [tf]

    if type(color) is not list:
        color = [color]

    assert len(t0) == len(tf), f"{len(t0)=} and {len(tf)=}; they must have same shape"

    y_polar_out = []

    for ii, (start, stop) in enumerate(zip(t0, tf)):
        s0, sf = np.searchsorted(t, [start, stop])
        phase_bins, y_polar = weighted_histogram(
            x[s0:sf], y[s0:sf], bins=bins, wrap=True
        )
        y_polar_out.append(y_polar)

    y_polar_out = np.vstack(y_polar_out)
    m = np.mean(y_polar_out, 0)

    # Plotting
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(projection="polar")
    else:
        f = None

    if multi == "sem":
        lb = m - np.nanstd(y_polar_out, 0) / np.sqrt(y_polar_out.shape[0])
        ub = m + np.nanstd(y_polar_out, 0) / np.sqrt(y_polar_out.shape[0])
        ax.plot(phase_bins, np.mean(y_polar_out, 0), color=color[0], **plot_kwargs)
        ax.fill_between(phase_bins, lb, ub, color=color[0], alpha=alpha)
    elif multi == "std":
        lb = m - np.nanstd(y_polar_out, 0)
        ub = m + np.nanstd(y_polar_out, 0)
        ax.plot(phase_bins, np.mean(y_polar_out, 0), color=color[0], **plot_kwargs)
        ax.fill_between(phase_bins, lb, ub, color=color[0], alpha=alpha)
    else:
        for ii, y_polar in enumerate(y_polar_out):
            if len(color) == 1:
                c = color[0]
            else:
                c = color[ii]
            ax.plot(phase_bins, y_polar, color=c, **plot_kwargs)
    clean_polar_axis(ax)
    return (f, ax, y_polar_out, phase_bins)


def plot_reset_curve(
    breaths,
    events,
    wavelength=473,
    annotate=False,
    norm=True,
    plot_tgl=True,
    n_control=100,
):
    """
    Plot a reset curve for optogenetic stimulation, showing phase-dependent effects on breathing cycles.

    Args:
        breaths (AlfBunch): Breath timing data with attributes: 'times', 'IBI', and 'duration_sec'.
        events (np.ndarray): 1D array of stimulation/event times.
        wavelength (int, optional): Wavelength of optogenetic stimulus. Defaults to 473.
        annotate (bool, optional): If True, add annotations and color overlays to the plot. Defaults to False.
        norm (bool, optional): If True, normalizes time to phase (0-1) for plotting. Defaults to True.
        plot_tgl (bool, optional): If True, creates a plot; if False, returns computed data. Defaults to True.
        n_control (int, optional): Number of random control points for a control distribution. Defaults to 100.

    Returns:
        tuple:
            - `cycle_stim_time`: Normalized/raw times of stimulation relative to breath onset.
            - `cycle_duration`: Normalized/raw breath cycle durations following stimulation.
            - `cycle_stim_time_rand`: Control times for stimulation from random event times.
            - `cycle_duration_rand`: Control breath cycle durations for random events.
    """

    def _get_relative_times(times, events):
        idx_last = np.searchsorted(times, events) - 1
        idx_next = idx_last + 1
        return events - times[idx_last], times[idx_next] - events

    # Filter breaths within the event range
    t0, tf = events.min(), events.max()
    valid_breaths = (breaths.times > t0) & (breaths.times < tf)
    mean_IBI, mean_dur = (
        breaths.IBI[valid_breaths].mean(),
        breaths.duration_sec[valid_breaths].mean(),
    )

    norm_value = mean_IBI if norm else 1

    # Calculate random control data
    rand_samp = np.random.uniform(low=t0, high=tf, size=n_control)
    t_last_rand, t_next_rand = _get_relative_times(breaths.times, rand_samp)
    cycle_duration_rand = (t_next_rand + t_last_rand) / norm_value
    cycle_stim_time_rand = t_last_rand / norm_value

    # Calculate stimulation event data
    t_last, t_next = _get_relative_times(breaths.times, events)
    cycle_duration = (t_next + t_last) / norm_value
    cycle_stim_time = t_last / norm_value

    # Plot control data
    if plot_tgl:
        plt.plot(
            cycle_stim_time_rand, cycle_duration_rand, "ko", ms=3, alpha=0.5, mew=0
        )

    # Plot stimulation event data
    if plot_tgl:
        plt.plot(
            cycle_stim_time,
            cycle_duration,
            "o",
            color=laser_colors[wavelength],
            mec="k",
            mew=0,
        )

    # Return computed data if plot is disabled
    if not plot_tgl:
        return (
            cycle_stim_time,
            cycle_duration,
            cycle_stim_time_rand,
            cycle_duration_rand,
        )

    # Plot aesthetics
    def _prettify_plot(norm, mean_dur, mean_IBI):
        if norm:
            plt.axvline(mean_dur / mean_IBI, color="k", ls="--", lw=0.5)
            plt.axhline(1, color="k", ls="--", lw=0.5)
            plt.plot([0, 2], [0, 2], color="tab:red")
            plt.xlabel("Stim time (normalized)")
            plt.ylabel("Cycle duration (normalized)")
            plt.xlim(0, 1.5)
            plt.ylim(0, 2)
            plt.xticks([0, 0.5, 1])
            plt.yticks([0, 1, 2])
        else:
            xmax = np.max(np.concatenate([t_last, t_last_rand]))
            ymax = np.max(np.concatenate([t_next, t_next_rand]))
            plt.axvline(mean_dur, color="k", ls="--", lw=0.5)
            plt.axhline(mean_IBI, color="k", ls="--", lw=0.5)
            plt.plot(
                [0, mean_dur + mean_IBI], [0, mean_IBI + mean_dur], color="tab:red"
            )
            plt.xlabel("Time since last breath onset (s)")
            plt.ylabel("Total time between breaths (s)")
            plt.xlim([0, xmax])
            plt.ylim([0, ymax * 1.1])

    _prettify_plot(norm, mean_dur, mean_IBI)

    # Add annotations and overlays if requested
    if annotate:

        def _add_annotations(mean_dur, mean_IBI):
            plt.text(
                0.01, 1.5, "Prolong inspiration", ha="left", va="bottom", rotation=90
            )
            plt.text(
                0.01, 0.01, "Shorten inspiration", ha="left", va="bottom", rotation=90
            )
            plt.text(
                mean_dur / mean_IBI + 0.01,
                mean_dur / mean_IBI + 0.05,
                "Phase advance",
                rotation=90,
            )
            plt.text(mean_dur / mean_IBI + 0.01, 1.5, "Phase delay", rotation=90)

            plt.fill_between(
                [0, mean_dur / mean_IBI],
                [0, mean_dur / mean_IBI],
                [1, 1],
                color="tab:purple",
                alpha=0.2,
            )
            plt.fill_between(
                [0, mean_dur / mean_IBI], [1, 1], [2, 2], color="tab:green", alpha=0.2
            )

            pts = np.array(
                [
                    [mean_dur / mean_IBI, 1],
                    [1, 1],
                    [1.5, 1.5],
                    [1.5, 2],
                    [mean_dur / mean_IBI, 2],
                ]
            )
            plt.fill(pts[:, 0], pts[:, 1], color="tab:orange", alpha=0.2)
            plt.fill_between(
                [mean_dur / mean_IBI, 1],
                [mean_dur / mean_IBI, 1],
                [1, 1],
                color="tab:grey",
                alpha=0.2,
            )

            plt.text(
                mean_dur / mean_IBI / 2,
                mean_dur / mean_IBI / 2 * 0.8,
                "Lower bound",
                color="tab:red",
                rotation=26,
            )
            plt.text(
                mean_dur / mean_IBI / 2,
                plt.gca().get_ylim()[1],
                "Inspiration",
                ha="center",
                va="top",
            )
            plt.text(
                mean_dur / mean_IBI + (1 - mean_dur / mean_IBI) / 2,
                plt.gca().get_ylim()[1],
                "Expiration",
                ha="center",
                va="top",
            )

        _add_annotations(mean_dur, mean_IBI)

    sns.despine()
    return cycle_stim_time, cycle_duration, cycle_stim_time_rand, cycle_duration_rand


def plot_sweeps(xt, x, times, pre, post, ax=None, **kwargs):
    """
    Time-aligns a trace `x` to event times specified in `times`.

    Args:
        xt (array-like): Time values corresponding to the signal trace `x`.
        x (array-like): Signal trace data to be plotted.
        times (array-like): Event times to align the trace `x` to.
        pre (float): Time before each event to start the trace.
        post (float): Time after each event to end the trace.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure and axes will be created. Default is None.
        **kwargs: Additional keyword arguments passed to `ax.plot`, such as line style or color.

    Returns:
        ax: matplotlib axis
    """
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot()
    for tt in times:
        t0 = tt - pre
        tf = tt + post
        s0, st, sf = np.searchsorted(xt, [t0, tt, tf])-1
        ax.plot(xt[s0:sf] - xt[st], x[s0:sf], **kwargs)
    return ax


def plot_most_likely_dynamics(
    model,
    xlim=(-4, 4),
    ylim=(-3, 3),
    nxpts=20,
    nypts=20,
    alpha=0.8,
    ax=None,
    figsize=(3, 3),
    colors=[f"C{x}" for x in range(7)],
    zval=None
):
    """
    Plotting of underlying vector fields from Linderman Lab
    """
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(
                xy[zk, 0],
                xy[zk, 1],
                dxydt_m[zk, 0],
                dxydt_m[zk, 1],
                color=colors[k % len(colors)],
                alpha=alpha,
            )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    return ax


def plot_most_likely_dynamics_3D(
    model,
    xlim=(-4, 4),
    ylim=(-3, 3),
    zlim=(-3, 3),
    nxpts=10,
    nypts=10,
    nzpts=10,
    alpha=0.2,
    ax=None,
    figsize=(3, 3),
    length=0.2,
    colors=[f"C{x}" for x in range(7)],
):
    """
    Extension of the linderman vectorfield plot to 3D

    """
    assert model.D == 3
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    z = np.linspace(*zlim, nzpts)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    xyz = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Get the probability of each state at each xyz location
    k_state = np.argmax(xyz.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxyzdt_m = xyz.dot(A.T) + b - xyz

        zk = k_state == k
        if zk.sum(0) > 0:
            ax.quiver(
                xyz[zk, 0],
                xyz[zk, 1],
                xyz[zk, 2],
                dxyzdt_m[zk, 0],
                dxyzdt_m[zk, 1],
                dxyzdt_m[zk, 2],
                color=colors[k % len(colors)],
                alpha=alpha,
                length=length,
            )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.grid(visible=False)
    plt.tight_layout()

    return ax


def _clean_3d_axes(ax, title, dims, pane_color, lims=None):
    """Modify 3D axes to be cleaner:
            Set title
            set axis labels
            make limits equal
            turn off grid
            set background color
    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes object to customize.
        title (str): The title of the plot.
        dims (tuple or list of ints): Dimensions to label the axes, corresponding to the 3D data dimensions (e.g., (0, 1, 2) for first three components).
        pane_color (tuple or None): RGB color to set for the panes (background of each axis). Use None for default color.
        lims (tuple or list of floats,optional): If None, autoscales axes. Axis limits to set for x, y, and z axes (e.g., (-1, 1) to set limits for all axes).

    Returns:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The modified axes object.

    Example:
        ax = fig.add_subplot(111, projection='3d')
        _clean_3d_axes(ax, "3D Plot", (0, 1, 2), (0.9, 0.9, 0.9, 0.5), (-1, 1))
    """
    ax.set_title(title)

    if lims is not None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_zlim(lims)

    ax.set_xlabel(f"Dim {dims[0]+1}")
    ax.set_ylabel(f"Dim {dims[1]+1}")
    ax.set_zlabel(f"Dim {dims[2]+1}")

    ax.grid(False)

    if pane_color is not None:
        ax.xaxis.set_pane_color(pane_color)  # Set the color of the x-axis pane
        ax.yaxis.set_pane_color(pane_color)  # Set the color of the y-axis pane
        ax.zaxis.set_pane_color(pane_color)  # Set the color of the z-axis pane
    return ax


def clean_polar_axis(ax):
    """
    Clean the appearance of a polar plot.
    Use pi/2 (90 degrees) angular ticks, no internal radial ticks, and set labels to mathtext pi

    Args:
        ax (matplotlib.projections.polar.PolarAxes): The polar axes object to modify.

    Example:
        ax = plt.subplot(projection='polar')
        clean_polar_axis(ax)
    """
    ax.set_yticks([ax.get_yticks()[-1]])
    ax.set_xticks([0, np.pi / 2, np.pi, np.pi * 3 / 2])
    ax.set_xticklabels(["0", "$\\frac{\pi}{2}$", "$\pi$", "$\\frac{-\pi}{2}$"])


def clean_linear_radial_axis(ax):
    """
    Clean the appearance of a plot with a range o [-pi,pi] but on a normal, linear axis
    Sets ticks to every pi/2 interval and uses math text.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The axes object to modify.

    Example:
        ax = plt.subplot()
        clean_linear_radial_axis(ax)
    """
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(
        ["$-\pi$", "$\\frac{-\pi}{2}$", "0", "$\\frac{\pi}{2}$", "$\pi$"]
    )
    sns.despine(trim=True)


def plot_driftmap_with_trace(
    spike_times,
    spike_depths,
    trace,
    trace_times,
    trace_label="",
    t0=None,
    tf=None,
    depth_lim=(None, None),
    trace_ylim=(None, None),
    t_bin=0.01,
    driftmap_kwargs={},
    trace_kwargs={},
    figsize=(2, 8),
    use_scalebar=True,
    use_colorbar=True,
    cmap=None,
    raster_ylabel=None,
):
    """
    #TODO: Update documentation
    #TODO: UPdate to work more intuitively with clusters

    Works well as a driftmap, but lass good as a rastermap


    Plot a drift map with an covariate trace above.


    Built off of ibllib.brainbox.plot.driftmap.
    This function plots a drift map of spike times and depths, with an overlaid trace such as diaphragm or another continuous signal.

    Args:
        spike_times (array-like): Array of spike times.
        spike_depths (array-like): Array of spike depths corresponding to each spike time.
        trace (array-like): Continuous signal to overlay on the drift map. Can be multiple columns as long as each row is a timepoint
        trace_times (array-like): Time points corresponding to the trace signal.
        trace_label (str, optional): Label for the trace. Defaults to ''.
        t0 (float, optional): Start time for the plot. Defaults to None, which uses the minimum spike time.
        tf (float, optional): End time for the plot. Defaults to None, which uses the maximum spike time.
        depth_lim (tuple, optional): Depth limits for the plot. Defaults to (None, None), which uses the min and max spike depths.
        trace_ylim (tuple, optional): Y-axis limits for the trace plot. Defaults to (None, None), which uses the min and max trace values.
        driftmap_kwargs (dict, optional): Additional keyword arguments for the drift map plot. Defaults to {}.
        trace_kwargs (dict, optional): Additional keyword arguments for the trace plot. Defaults to {}.
        figsize (tuple, optional): Figure size for the plot. Defaults to (2, 8).
        use_scalebar (bool, optional): Whether to use a scale bar in the plot. Defaults to True.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The axes object containing the drift map.
        matplotlib.axes._subplots.AxesSubplot: The axes object containing the trace plot.
    """
    assert len(depth_lim) == 2, "depth_lim must be of length 2"
    assert len(trace_ylim) == 2, "trace_ylim must be of length 2"

    # set default trace kwargs and overwrite with user input
    trace_kwargs_default = {"lw": 0.5}
    trace_kwargs_default.update(trace_kwargs)
    trace_kwargs = trace_kwargs_default

    # set default driftmap kwargs and overwrite with user input
    driftmap_kwargs_default = {}
    driftmap_kwargs_default.update(driftmap_kwargs)
    driftmap_kwargs = driftmap_kwargs_default

    # Set time limits for spikes
    t0 = t0 or 0
    tf = tf or np.nanmax(spike_times)
    s0, sf = np.searchsorted(spike_times, [t0, tf])

    # Set depth limits
    depth_lim = list(depth_lim)
    depth_lim[0] = depth_lim[0] or np.min(spike_depths)
    depth_lim[1] = depth_lim[1] or np.max(spike_depths)

    # Subsample trace
    trace_samples = np.logical_and(trace_times >= t0, trace_times <= tf)
    trace_subset = trace[trace_samples]
    trace_times_subset = trace_times[trace_samples]

    trace_ylim = list(trace_ylim)
    trace_ylim[0] = trace_ylim[0] or np.min(trace_subset)
    trace_ylim[1] = trace_ylim[1] or np.max(trace_subset)

    # Make plot
    f = plt.figure(figsize=figsize)
    gs = f.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 5])
    ax_raster = f.add_subplot(gs[1])
    ax_trace = f.add_subplot(gs[0], sharex=ax_raster)
    ax_trace.plot(trace_times_subset, trace_subset, **trace_kwargs)
    driftmap(
        spike_times[s0:sf],
        spike_depths[s0:sf],
        ax=ax_raster,
        t_bin=t_bin,
        **driftmap_kwargs,
    )
    ax_raster.set_ylim(depth_lim)
    ax_trace.set_ylim(trace_ylim)
    ax_trace.set_ylabel(trace_label)
    print(ax_raster.get_ylim())
    sns.despine()
    if cmap is not None:
        ax_raster.images[0].set_cmap(cmap)

    # Add colorbar
    if use_colorbar:
        cbar_ax = f.add_axes([0.92, 0.25, 0.05, 0.25])  # [left, bottom, width, height]
        cbar = f.colorbar(ax_raster.images[0], cax=cbar_ax, orientation="vertical")
        cbar_ax.set_ylabel("spike rate (sp/s)")
        cbar.outline.set_linewidth(0)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/t_bin:0.0f}"))
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # replace x spine with a horizontal bar for for raster. Horizontal bar should be closeste to [1,2,5,10,30,60]
    if use_scalebar:
        replace_timeaxis_with_scalebar(ax_raster)
    plt.subplots_adjust(hspace=figsize[1] * 0.01)

    if raster_ylabel is not None:
        ax_raster.set_ylabel(raster_ylabel)

    return (ax_raster, ax_trace)


def replace_timeaxis_with_scalebar(ax, pad=0.025,lw=None,color=None,size=None):
    """
    Replace the x-axis with a horizontal bar showing the time scale of the plot.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The axes object to modify.
        inverted_y (bool, optional): Set true if the plot has 0 at the top
        pad (float, optional): Padding between the scale bar and the plot. Defaults to 0.01.
        lw (float, optional): Line width for the scale bar. Defaults to None, which uses the default line width.
        color (str, optional): Color for the scale bar. Defaults to None, which uses the default text color.
        size (float, optional): Font size for the scale bar. Defaults to None, which uses the default tick size.
    """
    if lw is None:
        lw = plt.rcParams['axes.linewidth']
    if color is None:
        color = plt.rcParams['text.color']
    if size is None:
        size = plt.rcParams['xtick.labelsize']


    t0, tf = ax.get_xlim()
    good_tbars = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60]
    idx = np.searchsorted(good_tbars, (tf - t0) / 5) - 1
    tbar_max = t0 + good_tbars[idx]
    tbar_length = tbar_max - t0
    tbar_label = (
        f"{tbar_length*1000:0.0f}ms" if tbar_length < 1 else f"{tbar_length:.0f}s"
    )
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    pad_small = pad * yrange
    pad_large = pad_small * 1.1

    ax.set_ylim([ymin - pad_large, ymax])
    ax.hlines(ymin - pad_small, t0, tbar_max, lw=lw, color=color)
    ax.text(
        t0,
        ymin - pad_large,
        tbar_label,
        va="top",
        ha="left",
        color=color,
        size=size,
    )
    ax.set_xticks([])
    ax.set_xlabel("")
    sns.despine(bottom=True)


def trim_yscale_to_lims(ax, ymin, ymax):
    ax.set_yticks([ymin, ymax])
    sns.despine(bottom=True, trim=True)


if has_brainbox:

    def plot_peth_and_raster(
        spike_times,
        starts,
        stops=None,
        pre_time=0.2,
        post_time=0.2,
        bin_size=0.01,
        smoothing=0,
        error_bars="sem",
        pethline_kwargs={},
        errbar_kwargs={"alpha": 0.5},
        eventline_kwargs={"color": plt.rcParams["text.color"], "ls": "--"},
        raster_kwargs={"s": 2, "marker": "|"},
        raster_ylabel="",
        figsize=(3, 6),
        subplot_ratio=(1, 5),
    ):
        """
        Plot a peri-event time histogram and raster plot

        Args:
            spike_times (array): spike times
            starts (array): event times
            stops (array, optional): stop times. Defaults to None.
            pre_time (float, optional): time before event to plot. Defaults to 0.2.
            post_time (float, optional): time after event to plot. Defaults to 0.2.
            bin_size (float, optional): bin size for histogram. Defaults to 0.01.
            smoothing (float, optional): smoothing factor for histogram. Defaults to 0.
            error_bars (str, optional): error bars to plot. Defaults to "sem".
            pethline_kwargs (dict, optional): kwargs for the peth line. Defaults to {}.
            errbar_kwargs (dict, optional): kwargs for the error bars. Defaults to {"alpha": 0.5}.
            eventline_kwargs (dict, optional): kwargs for the event line. Defaults to {"color": plt.rcParams["text.color"], "ls": "--"}.
            raster_kwargs (dict, optional): kwargs for the raster plot. Defaults to {'s':2,'marker':'|'}.
            raster_ylabel (str, optional): ylabel for the raster plot. Defaults to "".
            figsize (tuple, optional): figure size. Defaults to (3, 6).
            subplot_ratio (tuple, optional): ratio of the subplots. Defaults to (1, 5).

        Returns:
            ax_raster, ax_peth: matplotlib axis for the raster and peth plots
        """
        peth, _ = bbsc.calculate_peths(
            spike_times,
            np.ones_like(spike_times),
            [1],
            starts,
            pre_time=pre_time,
            post_time=post_time,
            bin_size=bin_size,
            smoothing=smoothing,
        )

        # Compute spikes separately for raster for better res
        peth_high_res, spikes = bbsc.calculate_peths(
            spike_times,
            np.ones_like(spike_times),
            [1],
            starts,
            pre_time=pre_time,
            post_time=post_time,
            bin_size=0.001,
            smoothing=smoothing,
        )

        duration_mean = None
        if stops is not None:
            assert len(starts) == len(stops)
            duration_mean = np.mean(stops - starts)

        use_error = True
        if error_bars == "sem":
            sem = peth["stds"][0] / np.sqrt(len(starts))
            lb = peth["means"][0] - sem
            ub = peth["means"][0] + sem
        elif error_bars == "std":
            lb = peth["means"][0] - peth["stds"][0]
            ub = peth["means"][0] + peth["stds"][0]
        else:
            use_error = False

        f = plt.figure(figsize=figsize)
        gs = f.add_gridspec(nrows=2, height_ratios=subplot_ratio)
        ax_raster = f.add_subplot(gs[1:, :])
        ax_peth = f.add_subplot(gs[0, :], sharex=ax_raster)

        ax_peth.plot(peth["tscale"], peth["means"][0], **pethline_kwargs)
        if use_error:
            ax_peth.fill_between(peth["tscale"], lb, ub, **errbar_kwargs)

        cc = peth_high_res["tscale"][np.where(spikes)[2]]
        rr = np.where(spikes)[0]
        ax_raster.scatter(cc, rr, **raster_kwargs)

        ax_peth.set_ylabel("FR (sp/s)")
        ax_peth.set_ylim([0, None])
        ax_raster.set_ylabel(raster_ylabel)
        ax_raster.set_xlabel("Time (s)")
        ax_raster.set_ylim([0, len(starts)])
        ax_peth.set_xlim([-pre_time, post_time])

        ax_raster.axvline(0, **eventline_kwargs)
        ax_peth.axvline(0, **eventline_kwargs)

        if duration_mean is not None:
            ax_peth.axvline(
                duration_mean, color=plt.rcParams["text.color"], linestyle="--"
            )
            ax_raster.axvline(
                duration_mean, color=plt.rcParams["text.color"], linestyle="--"
            )

        plt.tight_layout()

        return (ax_raster, ax_peth)

    def plot_multicondition_peth(
        spike_times,
        event_times,
        conditions,
        pre_time=0.2,
        post_time=0.2,
        bin_size=0.01,
        smoothing=0,
        error_bars="sem",
        pethline_kwargs={},
        errbar_kwargs={"alpha": 0.5},
        eventline_kwargs={"color": plt.rcParams["text.color"], "ls": "--"},
        ax=None,
        colors=None,
    ):
        """
            Plot a peri-event time histogram for multiple conditions

            Args:
                spike_times (array): spike times
                event_times (array): event times
                conditions (array): condition labels
                pre_time (float, optional): time before event to plot. Defaults to 0.2.
                post_time (float, optional): time after event to plot. Defaults to 0.2.
                bin_size (float, optional): bin size for histogram. Defaults to 0.01.
                smoothing (float, optional): smoothing factor for histogram. Defaults to 0.
                error_bars (str, optional): error bars to plot. Defaults to "sem".
                pethline_kwargs (dict, optional): kwargs for the peth line. Defaults to {}.
                errbar_kwargs (dict, optional): kwargs for the error bars. Defaults to {"alpha": 0.5}.
                eventline_kwargs (dict, optional): kwargs for the event line. Defaults to {"color": plt.rcParams["text.color"], "ls": "--"}.
                ax (matplotlib axis, optional): axis to plot on. Defaults to None.
                colors (list or dict, optional): colors for each condition. Defaults to None.

        Returns:
            matplotlib axis: axis with the plot

        """

        if ax is None:
            f, ax = plt.subplots(1, 1)

        unique_conditions = list(set(conditions))

        # Assign colors
        if colors is not None:
            if type(colors) is list:
                assert len(colors) == len(
                    set(conditions)
                ), "Number of colors must match number of conditions"
                color_map = {c: col for c, col in zip(unique_conditions, colors)}
            elif type(colors) is dict:
                color_map = colors
            else:
                raise ValueError("Colors must be a list or dict")
        else:
            color_map = {c: f"C{i}" for i, c in enumerate(unique_conditions)}

        # Remove color from kwargs if present
        pethline_kwargs.pop("color", None)
        errbar_kwargs.pop("color", None)

        assert (
            len(conditions) == len(event_times)
        ), f"Number of conditions {len(conditions)} must match number of event times {len(event_times)}"
        lines = []
        for condition in unique_conditions:
            idx = np.where(conditions == condition)[0]
            peth, _ = bbsc.calculate_peths(
                spike_times,
                np.ones_like(spike_times),
                [1],
                event_times[idx],
                pre_time=pre_time,
                post_time=post_time,
                bin_size=bin_size,
                smoothing=smoothing,
            )

            if error_bars == "sem":
                sem = peth["stds"][0] / np.sqrt(len(event_times[idx]))
                lb = peth["means"][0] - sem
                ub = peth["means"][0] + sem
            elif error_bars == "std":
                lb = peth["means"][0] - peth["stds"][0]
                ub = peth["means"][0] + peth["stds"][0]

            ll = ax.plot(
                peth["tscale"],
                peth["means"][0],
                label=condition,
                color=color_map[condition],
                **pethline_kwargs,
            )
            lines.append(ll[0])
            ax.fill_between(
                peth["tscale"], lb, ub, color=color_map[condition], **errbar_kwargs
            )
        ax.axvline(0, **eventline_kwargs)
        ax.set_ylabel("FR (sp/s)")
        ax.set_xlabel("Time (s)")
        ax.legend(lines, unique_conditions)
        ax.set_xlim([-pre_time, post_time])
        ax.set_ylim([0, None])

        return ax
