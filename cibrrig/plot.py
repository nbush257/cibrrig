import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from .utils.utils import weighted_histogram, parse_opto_log, validate_intervals
from one.alf.io import AlfBunch
import seaborn as sns
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.cm import ScalarMappable


laser_colors = {473: "#00b7ff", 565: "#d2ff00", 635: "#ff0000"}


def plot_laser(laser_in, **kwargs):
    """
    kwargs:
    amplitudes
    ax
    mode (shade,bar,vline)
    amp_label
    wavelength
    alpha
    **plot kwargs
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


# TODO: Clean and refactor some of this line plotting.
# TODO: Wrap into population object.
def plot_projection_line_multicondition(
    X, tbins, intervals, colors, dims=[0, 1], ax=None, alpha=0.5,lw=1
):
    """
    overlay multiple conditions each with a defined color
    """
    validate_intervals(intervals[:, 0], intervals[:, 1], overlap_ok=True)
    assert len(colors) == intervals.shape[0]
    if ax is None:
        f = plt.figure()
        if len(dims) == 2:
            ax = f.add_subplot()
        elif len(dims) == 2:
            ax = f.add_subplot(projection="3d")
    for ii, cc in enumerate(colors):
        t0, tf = intervals[ii]
        s0, sf = np.searchsorted(tbins, [t0, tf])

        X_sub = X[s0 - 1 : sf, :]
        ax = plot_projection_line(
            X_sub, dims=dims, cvar=None, color=cc, alpha=alpha, ax=ax,lw=lw
        )


def plot_projection_line(X, cvar=None, dims=[0, 1], cmap="viridis", **kwargs):
    if len(dims) == 2:
        ax = _plot_projection_line_2D(X, cvar, dims, cmap=cmap, **kwargs)
    elif len(dims) == 3:
        ax = _plot_projection_line_3D(X, cvar, dims=dims, cmap=cmap,**kwargs)
    else:
        raise ValueError("Number of dims must be two or three")

    # # TODO Fix colorbar
    # if plot_colorbar:
    #     cax = plt.gcf().add_axes([0.25, 0.85, 0.5, 0.02])
    #     cbar = plt.gcf().colorbar(p,cax=cax,orientation='horizontal')
    #     cbar.set_label(colorbar_title)
    #     cbar.solids.set(alpha=1)
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
    colorbar_title='',
    **kwargs,
):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot()


    segments = np.stack([X[:-1,dims], X[1:,dims]], axis=1)

    if cvar is not None:
        vmin = vmin or np.min(cvar)
        vmax = vmax or np.max(cvar)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        lc = LineCollection(segments,alpha=alpha,lw=lw,cmap=cmap,**kwargs)
        lc.set_array(cvar)
        lc.set_norm(norm)
    else:
        lc = LineCollection(segments,alpha=alpha,lw=lw,**kwargs)
        lc.set_color(color)
    
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel(f'Dim {dims[0]+1}')
    ax.set_ylabel(f'Dim {dims[1]+1}')

    if cvar is not None:
        cbar = plt.colorbar(lc, ax=ax,pad=0.1,orientation='horizontal',location='top')
        cbar.set_label(colorbar_title)
        cbar.set_ticks([vmin,0,vmax])
        cbar.solids.set_alpha(1)


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
    lims=[-4, 4],
    pane_color=None,
    colorbar_title="",
    plot_colorbar=True,
    vmin=None,
    vmax=None,
    lw=0.5,
    **kwargs,
):

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(projection="3d")
    segments = np.stack([X[:-1,dims], X[1:,dims]], axis=1)

    if cvar is not None:
        vmin = vmin or np.min(cvar)
        vmax = vmax or np.max(cvar)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        colors = cmap(norm(cvar[:-1]))
        lc = Line3DCollection(segments,alpha=np.ones_like(cvar[:-1])*alpha,lw=lw,colors=colors,**kwargs)
        lc.set_norm(norm)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(colors)
        cbar = plt.colorbar(sm, ax=ax, pad=0.1,location='top',orientation='horizontal')
        cbar.set_label(colorbar_title)
        cbar.solids.set_alpha(1)
    else:
        lc = Line3DCollection(segments,alpha=alpha,lw=lw,**kwargs)
        lc.set_color(color)
    
    ax.add_collection(lc)
    ax.autoscale()

    _clean_3d_axes(ax, title, dims, pane_color, lims)

    return ax


def plot_projection(X, dims, **kwargs):
    if len(dims) == 2:
        return plot_2D_projection(X, dims, **kwargs)
    elif len(dims) == 3:
        return plot_3D_projection(X, dims, **kwargs)
    else:
        raise (ValueError(f"Number of plotted dimensions must be 2 or 3. {dims=}"))


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
    lims=[-4, 4],
    plot_colorbar=True,
    colorbar_title="",
    pane_color=None,
    **kwargs
):
    assert len(dims) == 3, f"Must choose 3 dimensions to plot. Chose {dims}"
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111, projection="3d")
    else:
        f = ax.get_figure()

    assert isinstance(ax, Axes3D), "ax must be a 3D projection"

    if cvar is None:
        p = ax.scatter(
            X[:, dims[0]], X[:, dims[1]], X[:, dims[2]], c=c, s=s, alpha=alpha
        )
    else:
        vmin = vmin or np.min(cvar)
        vmax = vmax or np.max(cvar)
        p = ax.scatter(
            X[:, dims[0]],
            X[:, dims[1]],
            X[:, dims[2]],
            c=cvar,
            s=s,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            **kwargs
        )
        if plot_colorbar:
            cbar = plt.colorbar(p, ax=ax,pad=0.1,orientation='horizontal',location='top')
            cbar.set_label(colorbar_title)
            cbar.set_ticks([vmin,0,vmax])
            cbar.solids.set(alpha=1)

    _clean_3d_axes(ax, title, dims, pane_color, lims)

    return (f, ax)


def _clean_3d_axes(ax, title, dims, pane_color, lims):
    ax.set_title(title)

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
    return(ax)


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
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    else:
        f = ax.get_figure()
    assert len(dims) == 2, f"Must choose 2 dimensions to plot. Chose {dims}"

    if cvar is None:
        p = ax.scatter(X[:, dims[0]], X[:, dims[1]], color=c, s=s, alpha=alpha)
    else:
        vmin = vmin or np.min(cvar)
        vmax = vmax or np.max(cvar)
        p = ax.scatter(
            X[:, dims[0]],
            X[:, dims[1]],
            c=cvar,
            s=s,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
        if plot_colorbar:
            cbar = plt.colorbar(p, ax=ax,pad=0.1,orientation='horizontal',location='top')
            cbar.set_label(colorbar_title)
            cbar.set_ticks([vmin,0,vmax])
            cbar.solids.set(alpha=1)

    ax.set_title(title)

    ax.autoscale()
    ax.set_aspect('equal')

    ax.set_xlabel(f"Dim {dims[0]+1}")
    ax.set_ylabel(f"Dim {dims[1]+1}")
    ax.spines[["right", "top"]].set_visible(False)

    return (f, ax)


def clean_polar_axis(ax):
    ax.set_yticks([ax.get_yticks()[-1]])
    ax.set_xticks([0, np.pi / 2, np.pi, np.pi * 3 / 2])
    ax.set_xticklabels(["0", "$\\frac{\pi}{2}$", "$\pi$", "$\\frac{-\pi}{2}$"])


def clean_linear_radial_axis(ax):
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels(
        ["$-\pi$", "$\\frac{-\pi}{2}$", "0", "$\\frac{\pi}{2}$", "$\pi$"]
    )
    sns.despine(trim=True)


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
    """Plot y as a function of x on a polar scale.
    x should have domain [-pi,pi]
    t0 and tf can be times or lists of times. If left  None, the entire data will be considered
    if t0 and tf are lists or arrays, then  the average of the intervals defined by[t0_i,tf_i] will be computed, and the shaded region will be defined by "multi"

    # TODO: work with multiple conditions

    Args:
        x (1D numpy array): _description_
        y (1D numpy array): _description_
        t (1D numpy array): _description_
        ax (matplotlib axis, optional): _description_. Defaults to None.
        t0 (int or list, optional): start times of the epochs to consider. Defaults to None.
        tf (int or list, optional): end times of the epochs to consider. Defaults to None.
        color (str, optional): color. Defaults to 'k'.
        bins (int, optional): number of bins to split the unit circle into. Defaults to 50.
        multi (str, optional): What metrics to use for the shaded region. Can be ['std','sem']. If not these, then individual traces are plotted. Defaults to 'sem'.
        alpha (float, optional): Transparency of shaded region. Defaults to 0.3.
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
    Plot a reset curve. Designed for opto stimulation plotting
    TODO: Make general for times, not specifically a breaths alf object since it really only needs on and off timing
    TODO: maybe normalize to -pi->pi

    returns: (cycle_stim_time,cycle_duration,cycle_stim_time_rand,cycle_duration_rand)
    (x_stim,y_stim,x_control,y_control)


    Args:
        breaths (AlfBunch): Breath timing data. Needs attributes "times","IBI", and "duration_sec".
        events (1D numpy array): Times of events to compute the phase reset for
        wavelength (int, optional): Wavelength of the optical stimulus. Defaults to 473.
        annotate (bool, optional): Make annotations and colors on the plot to aid in understanding. Can be noisy. Defaults to False.
        norm (bool, optional): Whether to normalize the times to phase (0-1). Defaults to True.
        plot_tgl (bool, optional): Flag to include plotting. Defaults to True.
    """

    def _get_relative_times(times, events):
        idx_last = np.searchsorted(times, events) - 1
        idx_next = idx_last + 1
        t_last = events - times[idx_last]
        t_next = times[idx_next] - events
        return (t_last, t_next)

    t0, tf = events.min(), events.max()
    idx = np.logical_and(breaths.times > t0, breaths.times < tf)

    mean_IBI = breaths.IBI[idx].mean()
    mean_dur = breaths.duration_sec[idx].mean()
    if norm:
        norm_value = mean_IBI
    else:
        norm_value = 1
    xmax = []
    ymax = []
    rand_samp = np.random.uniform(low=t0, high=tf, size=(n_control,))

    # Compute and plot vals
    t_since_last_on_rand, t_to_next_on_end = _get_relative_times(
        breaths.times, rand_samp
    )
    cycle_duration_rand = (t_to_next_on_end + t_since_last_on_rand) / norm_value
    cycle_stim_time_rand = t_since_last_on_rand / norm_value
    if plot_tgl:
        (ctrls,) = plt.plot(
            cycle_stim_time_rand, cycle_duration_rand, "ko", ms=3, alpha=0.5, mew=0
        )

    t_since_last_on, t_to_next_on = _get_relative_times(breaths.times, events)
    cycle_duration = (t_to_next_on + t_since_last_on) / norm_value
    cycle_stim_time = t_since_last_on / norm_value
    if plot_tgl:
        (stims,) = plt.plot(
            cycle_stim_time,
            cycle_duration,
            "o",
            color=laser_colors[wavelength],
            mec="k",
            mew=0,
        )

    # # Skip plotting and just output data
    if not plot_tgl:
        return (
            cycle_stim_time,
            cycle_duration,
            cycle_stim_time_rand,
            cycle_duration_rand,
        )

    # Prettify plot
    if norm:
        # Essential plot features
        plt.axvline(mean_dur / mean_IBI, color="k", ls="--", lw=0.5)
        plt.axhline(1, color="k", ls="--", lw=0.5)
        plt.plot([0, 2], [0, 2], color="tab:red")
        plt.xlabel("Stim time (normalized)")
        plt.ylabel("Cycle duration (normalized)")
        plt.xlim(0, 1.5)
        plt.ylim(0, 2)
        plt.yticks([0, 1, 2])
        plt.xticks([0, 0.5, 1])

        # Accessory plot features
        if plot_tgl & annotate:
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
                ha="left",
                va="bottom",
                rotation=90,
            )
            plt.text(
                mean_dur / mean_IBI + 0.01,
                1.5,
                "Phase delay",
                ha="left",
                va="bottom",
                rotation=90,
            )

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
            # plt.fill_between([mean_dur / mean_IBI, 1], [1, 1], [2, 2], color='tab:orange', alpha=0.2)
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

            plt.xlim(0, 1.5)
            plt.ylim(0, 2)
            plt.yticks([0, 1, 2])
            plt.xticks([0, 0.5, 1, 1.5])
    else:
        xmax = np.max(np.concatenate([t_since_last_on, t_since_last_on_rand]))
        ymax = np.max(np.concatenate([t_to_next_on, t_to_next_on_end]))
        plt.axvline(mean_dur, color="k", ls="--", lw=0.5)
        plt.axhline(mean_IBI, color="k", ls="--", lw=0.5)

        plt.plot([0, mean_dur + mean_IBI], [0, mean_IBI + mean_dur], color="tab:red")

        plt.xlabel("Time since last breath onset (s)")
        plt.ylabel("Total time between breaths (s)")

        plt.xlim([0, xmax])
        plt.ylim([0, ymax * 1.1])

        # Phase advance
        pts = np.array(
            [[mean_dur, mean_dur], [mean_IBI, mean_IBI], [mean_dur, mean_IBI]]
        )
        plt.fill_between(pts[:, 0], pts[:, 1], color="tab:green", alpha=0.3)

        # Phase delay
        pts = np.array(
            [
                [mean_dur, mean_IBI],
                [mean_IBI, mean_IBI],
                [mean_IBI + mean_dur, mean_IBI + mean_dur],
                [mean_IBI + mean_dur, plt.gca().get_ylim()[1]],
                [mean_dur, plt.gca().get_ylim()[1]],
            ]
        )
        plt.fill(pts[:, 0], pts[:, 1], color="tab:grey", alpha=0.3)

        # Shorten inspiration
        pts = np.array(
            [[0, 0], [mean_dur, mean_dur], [mean_dur, mean_IBI], [0, mean_IBI]]
        )
        plt.fill(pts[:, 0], pts[:, 1], color="tab:purple", alpha=0.3)

        # Prolong inspiration
        pts = np.array(
            [
                [0, mean_IBI],
                [mean_dur, mean_IBI],
                [mean_dur, plt.gca().get_ylim()[1]],
                [0, plt.gca().get_ylim()[1]],
            ]
        )
        plt.fill(pts[:, 0], pts[:, 1], color="tab:orange", alpha=0.3)

        if annotate:
            plt.text(mean_dur, mean_dur / 2, "inspiration\nduration", rotation=90)
            plt.text(0, (mean_dur + mean_IBI) / 2, "Shorten inspiration", ha="left")
            plt.text(
                0, plt.gca().get_ylim()[1], "Prolong inspiration", ha="left", va="top"
            )
            plt.text(
                mean_dur,
                (mean_dur + mean_IBI) / 2,
                "Advance phase",
                ha="left",
                va="center",
            )
            plt.text(
                mean_dur, plt.gca().get_ylim()[1], "Delay phase", ha="left", va="top"
            )
            plt.text(
                (mean_dur + mean_IBI) / 2,
                (mean_dur + mean_IBI) / 2,
                "Lower bound",
                color="tab:red",
                ha="left",
                va="top",
            )
            plt.legend([stims, ctrls], ["Stims", "Random"])

    sns.despine()
    return (cycle_stim_time, cycle_duration, cycle_stim_time_rand, cycle_duration_rand)


def plot_sweeps(xt, x, times, pre, post, ax=None, **kwargs):
    """
    Time align a trace x to the event times in "times"
    """
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot()
    for tt in times:
        t0 = tt - pre
        tf = tt + post
        s0, st, sf = np.searchsorted(xt, [t0, tt, tf])
        ax.plot(xt[s0:sf] - xt[st], x[s0:sf], **kwargs)


def plot_most_likely_dynamics(
    model,
    xlim=(-4, 4),
    ylim=(-3, 3),
    nxpts=20,
    nypts=20,
    alpha=0.8,
    ax=None,
    figsize=(3, 3),
    colors=[f'C{x}' for x in range(7)],
):
    '''
    Plotting of underlying vector fields from Linderman Lab
    '''

    K = model.K
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
    colors=[f'C{x}' for x in range(7)],
):
    '''
    Extension of the linderman vectorfield plot to 3D

    '''
    assert model.D == 3
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    z = np.linspace(*zlim, nzpts)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    xyz = np.column_stack((X.ravel(), Y.ravel(),Z.ravel()))

    # Get the probability of each state at each xy location
    k_state = np.argmax(xyz.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111,projection='3d')

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
                length=length
            )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.grid(visible=False)
    plt.tight_layout()


    return(ax)


