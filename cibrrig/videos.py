import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

logging.basicConfig()
_log = logging.getLogger()
_log.setLevel(logging.INFO)

# TODO: be able to modify only parts of this
PROJECTION_KWARGS = dict(lw=0.5, alpha=0.2, color="C0", cmap="RdBu_r")
TRAIL_KWARGS = dict(lw=3, alpha=1, color="C1")
HISTORY_KWARGS = dict(color="C1", lw=0.75, alpha=0.75)
AUX_KWARGS = dict(lw=0.5, color="C1")


# TODO: Organize default arguments?
# TODO: Refactor to avoid repitition?
def make_aux_raster_projection_with_stims(
    pop,
    intervals,
    aux,
    aux_t,
    fn_out,
    stim_color,
    aux_label="",
    frame_step=0.01,
    duration=2,
    cmap="bone",
    vmax=0.5,
    baseline=10,
    fps=30,
    dpi=300,
    dims=[0, 1],
    lead_in=1,
    winsize=1,
    trail_length=0.05,
    figsize=(3, 8),
    azim_speed=0.1,
    elev_speed=0.1,
    rotation_delay=1,
    rotate=True,
    style="dark_background",
    projection_kwargs=PROJECTION_KWARGS,
    trail_kwargs=TRAIL_KWARGS,
    history_kwargs=HISTORY_KWARGS,
    aux_kwargs=AUX_KWARGS,
):
    plt.style.use(style)
    assert lead_in < duration, f"{lead_in=} must be shorter than {duration=}"
    assert (
        isinstance(intervals, np.ndarray) and intervals.shape[1] == 2
    ), "Intervals must be an n x 2 array"

    t0 = intervals[0, 0] - lead_in
    tf = t0 + duration

    if baseline == 0:
        baseline = 30
        projection_kwargs["alpha"] = 0
    # Set up figure and axes layout =====
    f = plt.figure(figsize=figsize, dpi=dpi)
    gs = f.add_gridspec(nrows=15, ncols=1)
    if len(dims) == 3:
        ax = f.add_subplot(gs[8:, :], projection="3d")
    else:
        ax = f.add_subplot(gs[8:, :])

    ax_raster = f.add_subplot(gs[1:8, :])
    ax_aux = f.add_subplot(gs[0, :], sharex=ax_raster)

    # Plot baseline
    if baseline > 0:
        ax = pop.plot_projection_line(
            dims=dims, t0=t0 - baseline, tf=t0, ax=ax, **projection_kwargs
        )
    # Plot trail================================
    s0, sf = np.searchsorted(pop.tbins, [t0 - trail_length, t0])
    # Plot thick line (current timepoints)
    if len(dims) == 3:
        (trail1,) = ax.plot(
            pop.projection[s0:sf, dims[0]],
            pop.projection[s0:sf, dims[1]],
            pop.projection[s0:sf, dims[2]],
            **trail_kwargs,
        )  # NB: the comma after trail1 is important.
    else:
        (trail1,) = ax.plot(
            pop.projection[s0:sf, dims[0]],
            pop.projection[s0:sf, dims[1]],
            **trail_kwargs,
        )  # NB: the comma after trail1 is important.

    # Plot thin line (all previous timepoints)
    s0, sf = np.searchsorted(pop.tbins, [t0, tf])
    segments = np.stack(
        [pop.projection[s0 : sf - 1, dims], pop.projection[s0 + 1 : sf, dims]], axis=1
    )
    if len(dims) == 3:
        history = Line3DCollection(segments, **history_kwargs)
    else:
        history = LineCollection(segments, **history_kwargs)
    ax.add_collection(history)
    history.set_color("none")

    # Plot Raster ============================
    cell_ids = np.arange(pop.cbins.shape[0])
    s0, sf = np.searchsorted(pop.tbins, [t0 - winsize, t0 + winsize])
    quad = ax_raster.pcolormesh(
        pop.tbins[s0:sf] - t0,
        cell_ids,
        pop.raster_smoothed[:, s0:sf],
        cmap=cmap,
        vmax=vmax,
    )
    ax_raster.axvline(0, color=plt.rcParams["text.color"], ls=":", lw=2)
    ax_raster.axis("off")

    # annotations
    sns.despine(trim=True)
    ax_raster.set_yticks([])
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_xticks([-winsize, 0, winsize])
    ax_raster.set_xlim([-winsize * 1.1, winsize])
    ax_raster.vlines(-winsize * 1.05, 0, 25, lw=3)
    ax_raster.text(
        winsize * -1.08, 0, "25 neurons", rotation=90, ha="right", va="bottom"
    )
    ax_raster.hlines(-cell_ids.shape[0] * 0.05, -winsize, -winsize + winsize / 5, lw=3)
    ax_raster.text(
        -winsize,
        -cell_ids.shape[0] * 0.075,
        f"{winsize/5*1000:0.0f}ms",
        ha="left",
        va="top",
    )

    # Plot aux =================================
    s0, sf = np.searchsorted(aux_t, [t0 - winsize, t0 + winsize])
    (dd,) = ax_aux.plot(aux_t[s0:sf] - t0, aux[s0:sf], **aux_kwargs)

    # annotations
    ax_aux.axis("off")
    ax_aux.axvline(0, ls=":", lw=2)
    s0, sf = np.searchsorted(aux_t, [t0, tf])
    yy = np.min(aux[s0:sf]), np.max(aux[s0:sf])
    ax_aux.set_ylim(yy[0], yy[1] * 1.1)
    ax_aux.text(
        ax_aux.get_xlim()[0],
        np.mean(yy),
        aux_label,
        rotation=90,
        ha="right",
        va="center",
        color=aux_kwargs["color"],
    )
    # layout
    ymax = ax_aux.get_ylim()[1]

    aux_stims = ax_aux.hlines(
        np.ones(intervals.shape[0]) * ymax * 0.9,
        intervals[:, 0] - t0,
        intervals[:, 1] - t0,
        color=stim_color,
        lw=4,
    )

    # Limits of the projection
    ax = _trim_axes(ax, pop, t0, tf, dims)
    if len(dims) == 3:
        _plot_xy_plane(ax, color=plt.rcParams["text.color"], alpha=0.25)

    # INITIALIZE
    def init():
        return (trail1,)

    # UPDATE - This is the main loop to update each frame of the video. Using the "set_data" method will update the image object with new data for the new frame
    def update(frames):
        this_t = t0 + frames
        is_stim = False
        if np.any(np.logical_and(intervals[:, 0] < this_t, intervals[:, 1] > this_t)):
            is_stim = True

        # Update the trajectory (current time points)
        s0, sf = np.searchsorted(
            pop.tbins, [t0 + frames - trail_length, t0 + frames]
        )  # Use "frames" to get a new slice into the data (i.e., maps time into samples)
        trail1.set_data(pop.projection[s0:sf, dims[0]], pop.projection[s0:sf, dims[1]])
        if len(dims) == 3:
            trail1.set_3d_properties(pop.projection[s0:sf, dims[2]])

        # update the data
        if is_stim:
            trail1.set_color(stim_color)
        else:
            trail1.set_color(trail_kwargs["color"])  # Sets the color

        # Update the histroy trajectory (all previous time points)
        colors = np.array([trail_kwargs["color"]] * (segments.shape[0]), dtype="object")
        s0, sf = np.searchsorted(
            pop.tbins, [t0, tf]
        )  # Use "frames" to get a new slice into the data (i.e., maps time into samples)
        for t1, t2 in intervals:
            mask = (pop.tbins[s0 : sf - 1] >= t1) & (pop.tbins[s0 : sf - 1] <= t2)
            colors[mask] = stim_color
        history.set_color(colors)
        alphas = np.ones(segments.shape[0]) * history_kwargs["alpha"]
        invisible_mask = pop.tbins[s0 : sf - 1] > this_t
        alphas[invisible_mask] = 0
        history.set_alpha(alphas)

        # Get new samples for the raster image
        s0, sf = np.searchsorted(
            pop.tbins, [t0 + frames - winsize, t0 + frames + winsize]
        )
        C = pop.raster_smoothed[
            :, s0:sf
        ]  # Slice into the raser with the new datapoints
        quad.set_array(C.ravel())  # Update the plotted data

        # Get the new data for the diaphragm trace
        s0, sf = np.searchsorted(aux_t, [t0 + frames - winsize, t0 + frames + winsize])
        dd.set_data(aux_t[s0:sf] - t0 - frames, aux[s0:sf])  # Update with new slices

        # Rotate if 3D
        if len(dims) == 3 and rotate:
            if frames > rotation_delay:
                ax.view_init(ax.elev + elev_speed, ax.azim - azim_speed)

        # Update stim indicators
        for stim in aux_stims.get_paths():
            stim.vertices[:, 0] -= frame_step

        return trail1, history, quad, dd

    # This sets up the animation. Pass it "frames" which is a vector from zero to the total duration in desired time steps. "blit" may not always work but it should speed things up. Outside of my scope of knowledge
    ani = FuncAnimation(
        f, update, frames=np.arange(0, tf - t0, frame_step), init_func=init, blit=True
    )
    print(f"saving to {fn_out}")
    ani.save(fn_out, fps=fps, dpi=dpi)  # Performs and saves the animation.
    print("DONE!")


def _trim_axes(ax, pop, t0, tf, dims):
    ax.autoscale()
    ax.set_aspect("equal")
    s0, sf = np.searchsorted(pop.tbins, [t0, tf])
    xlim = (
        np.min(pop.projection[s0:sf, dims[0]]),
        np.max(pop.projection[s0:sf, dims[0]]),
    )
    ylim = (
        np.min(pop.projection[s0:sf, dims[1]]),
        np.max(pop.projection[s0:sf, dims[1]]),
    )
    if len(dims) == 2:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticks([xticks[0], 0, xticks[-1]])
        ax.set_yticks([yticks[0], 0, yticks[-1]])
        plt.tight_layout()
    elif len(dims) == 3:
        zlim = (
            np.min(pop.projection[s0:sf, dims[2]]),
            np.max(pop.projection[s0:sf, dims[2]]),
        )
        view_min = np.floor(np.min([xlim[0], ylim[0], zlim[0]]))
        view_max = np.ceil(np.max([xlim[1], ylim[1], zlim[1]]))
        ax.set_xlim([view_min, view_max])
        ax.set_ylim([view_min, view_max])
        ax.set_zlim([view_min, view_max])
        ax.set_xticks([view_min, 0, view_max])
        ax.set_yticks([view_min, 0, view_max])
        ax.set_zticks([view_min, 0, view_max])
        ax.get_proj()
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    sns.despine(trim=True)
    return ax


def _plot_xy_plane(ax, **kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Define the grid for the plane based on the current x and y limits
    x = np.linspace(xlim[0], xlim[1], 10)
    y = np.linspace(ylim[0], ylim[1], 10)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)

    # Plot the surface (plane) with transparency
    ax.plot_surface(x, y, z, rstride=100, cstride=100, **kwargs)


def _plot_all_planes(ax, **kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # Create grid for the XY plane
    x = np.linspace(xlim[0], xlim[1], 10)
    y = np.linspace(ylim[0], ylim[1], 10)
    x, y = np.meshgrid(x, y)
    z_xy = np.zeros_like(x)  # XY plane at z = 0

    # Create grid for the YZ plane
    y = np.linspace(ylim[0], ylim[1], 10)
    z = np.linspace(zlim[0], zlim[1], 10)
    y, z = np.meshgrid(y, z)
    x_yz = np.zeros_like(y)  # YZ plane at x = 0

    # Create grid for the XZ plane
    x = np.linspace(xlim[0], xlim[1], 10)
    z = np.linspace(zlim[0], zlim[1], 10)
    x, z = np.meshgrid(x, z)
    y_xz = np.zeros_like(x)  # XZ plane at y = 0

    # Plot the XY plane with transparency
    ax.plot_surface(x, y, z_xy, rstride=100, cstride=100, **kwargs)

    # Plot the YZ plane with transparency
    ax.plot_surface(x_yz, y, z, rstride=100, cstride=100, **kwargs)

    # Plot the XZ plane with transparency
    ax.plot_surface(x, y_xz, z, rstride=100, cstride=100, **kwargs)


def make_projection(
    pop,
    t0,
    duration,
    fn_out,
    stim_color,
    intervals=None,
    cvar=None,
    cvar_label="",
    cmap="magma",
    dims=[0, 1],
    frame_step=0.01,
    fps=30,
    dpi=300,
    trail_length=0.05,
    figsize=(4, 4),
    style="dark_background",
    projection_kwargs=PROJECTION_KWARGS,
    history_kwargs=HISTORY_KWARGS,
    trail_kwargs=TRAIL_KWARGS,
    baseline=20,
    mode="line",
    rotate=True,
    rotation_delay=1,
    elev_speed=0.2,
    azim_speed=0.2,
    vmin=None,
    vmax=None,
):
    plt.style.use(style)
    tf = t0 + duration
    f = plt.figure(figsize=figsize)
    is_3D = False
    if baseline == 0:
        baseline = 30
        projection_kwargs["alpha"] = 0
    if len(dims) == 3:
        is_3D = True

    if is_3D:
        ax = f.add_subplot(111, projection="3d")
    else:
        ax = f.add_subplot(111)

    if mode == "line":
        ax = pop.plot_projection_line(
            dims=dims,
            t0=t0 - baseline,
            tf=t0,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cvar=cvar,
            colorbar_title=cvar_label,
            **projection_kwargs,
        )
    elif mode == "scatter":
        if cvar is not None:
            projection_kwargs.pop("lw")
            projection_kwargs.pop("color")
            ax = pop.plot_projection(
                dims=dims,
                t0=t0 - baseline,
                tf=t0,
                ax=ax,
                cvar=cvar,
                vmin=vmin,
                vmax=vmax,
                colorbar_title=cvar_label,
                **projection_kwargs,
            )[1]
        else:
            ax = pop.plot_projection(
                dims=dims, t0=t0 - baseline, tf=t0, ax=ax, c=projection_kwargs["color"]
            )[1]
    else:
        raise ValueError('Mode must be "line" or "scatter')

    # Plot trail
    s1, s2 = np.searchsorted(pop.tbins, [t0 - trail_length, t0])
    (trail1,) = ax.plot(
        pop.projection[s1:s2, dims[0]],
        pop.projection[s1:s2, dims[1]],
        **trail_kwargs,
    )
    # Plot history================================
    s0, sf = np.searchsorted(pop.tbins, [t0, tf])
    segments = np.stack(
        [pop.projection[s0 : sf - 1, dims], pop.projection[s0 + 1 : sf, dims]], axis=1
    )
    if is_3D:
        history = Line3DCollection(segments, **history_kwargs)
    else:
        history = LineCollection(segments, **history_kwargs)
    ax.add_collection(history)

    # Set history color according to cvar
    if cvar is not None:
        vmin = vmin or np.min(cvar[s0 : sf - 1])
        vmax = vmax or np.max(cvar[s0 : sf - 1])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        history.set_color(cmap(norm(cvar[s0 : sf - 1])))
    else:
        history.set_color("none")

    # Clean up axes
    ax = _trim_axes(ax, pop, t0, tf, dims)

    # Add the XY plane in a 3D plot
    if is_3D:
        _plot_xy_plane(ax, color=plt.rcParams["text.color"], alpha=0.25)
        # ax.axis('off')
        # _plot_all_planes(ax,color=plt.rcParams['text.color'],alpha=0.25)

    def init():
        return (history,)

    def update(frames):
        this_t = t0 + frames

        #  Update history
        if cvar is None:
            colors = np.array(
                [trail_kwargs["color"]] * (segments.shape[0]), dtype="object"
            )
            if intervals is not None:
                for t1, t2 in intervals:
                    mask = (pop.tbins[s0 : sf - 1] >= t1) & (
                        pop.tbins[s0 : sf - 1] <= t2
                    )
                    colors[mask] = stim_color
            history.set_color(colors)

        alphas = np.ones(segments.shape[0]) * history_kwargs["alpha"]
        invisible_mask = pop.tbins[s0 : sf - 1] > this_t
        alphas[invisible_mask] = 0
        history.set_alpha(alphas)

        # Update trail
        s1, s2 = np.searchsorted(pop.tbins, [t0 + frames - trail_length, t0 + frames])
        trail1.set_data(pop.projection[s1:s2, dims[0]], pop.projection[s1:s2, dims[1]])
        if is_3D:
            trail1.set_3d_properties(pop.projection[s1:s2, dims[2]])

        # Color trail by stim
        is_stim = False
        if intervals is not None:
            if np.any(
                np.logical_and(intervals[:, 0] < this_t, intervals[:, 1] > this_t)
            ):
                is_stim = True
        if is_stim:
            trail1.set_color(stim_color)
        else:
            trail1.set_color(trail_kwargs["color"])  # Sets the color

        # Rotate if 3D
        if len(dims) == 3 and rotate:
            if frames > rotation_delay:
                ax.view_init(ax.elev + elev_speed, ax.azim - azim_speed)

        return (history,)

    ani = FuncAnimation(
        f, update, init_func=init, frames=np.arange(0, tf - t0, frame_step), blit=True
    )
    print(f"saving to {fn_out}")
    ani.save(fn_out, fps=fps, dpi=dpi)  # Performs and saves the animation.
    print("DONE!")


def make_rotating_projection(
    pop,
    t0,
    duration,
    fn_out,
    figsize=(4, 4),
    dims=[0, 1, 2],
    cvar=None,
    rotation_delay=1,
    elev_speed=0.1,
    azim_speed=0.1,
    mode="scatter",
    style="dark_background",
    cvar_label="",
    cmap="magma",
    vmin=None,
    vmax=None,
    frame_step=0.01,
    fps=30,
    dpi=300,
    n_frames=100,
    projection_kwargs=PROJECTION_KWARGS,
):
    plt.style.use(style)
    tf = t0 + duration
    assert len(dims) == 3, "Rotating projection does not make sense without 3D"
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111, projection="3d")
    (line,) = ax.plot(0, 1, ".", alpha=0)
    s0, sf = np.searchsorted(pop.tbins, [t0, tf])
    if cvar is not None:
        vmin = vmin or np.min(cvar[s0 : sf - 1])
        vmax = vmax or np.max(cvar[s0 : sf - 1])

    if mode == "line":
        projection_kwargs.pop("s")
        ax = pop.plot_projection_line(
            dims=dims,
            t0=t0,
            tf=tf,
            ax=ax,
            cvar=cvar,
            vmin=vmin,
            vmax=vmax,
            colorbar_title=cvar_label,
            **projection_kwargs,
        )
    elif mode == "scatter":
        projection_kwargs.pop("lw")
        projection_kwargs.pop("color")
        if cvar is not None:
            ax = pop.plot_projection(
                dims=dims,
                t0=t0,
                tf=tf,
                ax=ax,
                cvar=cvar,
                colorbar_title=cvar_label,
                vmin=vmin,
                vmax=vmax,
                **projection_kwargs,
            )[1]
        else:
            ax = pop.plot_projection(
                dims=dims, t0=t0, tf=tf, ax=ax, c=projection_kwargs["color"]
            )[1]
    else:
        raise ValueError('Mode must be "line" or "scatter')

    ax = _trim_axes(ax, pop, t0, tf, dims)
    _plot_xy_plane(ax, color=plt.rcParams["text.color"], alpha=0.25)

    s0, sf = np.searchsorted(pop.tbins, [t0, tf])

    def update(frames):
        # Rotate
        if frames > rotation_delay:
            ax.view_init(ax.elev + elev_speed, ax.azim - azim_speed)
        return (line,)

    ani = FuncAnimation(f, update, frames=np.arange(n_frames), blit=True)
    print(f"saving to {fn_out}")
    ani.save(fn_out, fps=fps, dpi=dpi)  # Performs and saves the animation.
    print("DONE!")


if __name__ == "__main__":
    # Testing:
    _log.info("Testing")
    from brainbox.io.one import SpikeSortingLoader
    from one.api import One

    from cibrrig.analysis.population import Population, get_good_spikes

    CACHE_DIR = "/data/hps/assoc/private/medullary/data/alf_data_repo"
    one = One(CACHE_DIR)
    eid = one.search(subject="m2024-40")[0]
    ssl = SpikeSortingLoader(one, eid=eid)
    spikes, clusters, channels = ssl.load_spike_sorting(spike_sorter="")

    spikes, cluster_ids = get_good_spikes(spikes, clusters)
    pop = Population(spikes.times, spikes.clusters)
    pop.compute_projection()
    physiology = one.load_object(eid, "physiology")
    dia = one.load_object(eid, "diaphragm")
    pdiff = pop.sync_var(physiology.pdiff, physiology.times)
    log = one.load_object(eid, "log").to_df()
    laser = one.load_object(eid, "laser").to_df()
    starts, stops = log.query('phase=="insp"')[["start_time", "end_time"]].values[0]
    intervals = laser.query("intervals_0>@starts and intervals_1<@stops")[
        ["intervals_0", "intervals_1"]
    ].values

    # intervals = np.array([[10.0, 11.0], [12.0, 13.0]])
    make_aux_raster_projection_with_stims(
        pop,
        intervals,
        dia.filtered,
        dia.times,
        "test.mp4",
        aux_label="dia",
        stim_color="c",
        lead_in=2,
        duration=4,
        dpi=100,
        dims=[0, 2],
        elev_speed=0.0,
        azim_speed=0.4,
        baseline=10,
    )

    trail_kwargs = TRAIL_KWARGS
    trail_kwargs["alpha"] = 0
    intervals = None

    # make_projection(
    #     pop,
    #     120,
    #     4,
    #     "test_projection.mp4",
    #     stim_color="r",
    #     intervals=intervals,
    #     dpi=300,
    #     figsize=(4, 4),
    #     frame_step=0.05,
    #     mode="scatter",
    #     dims=[0, 1],
    #     cvar=pdiff,
    #     cvar_label='Pressure',
    #     baseline=0,
    #     cmap="RdBu_r",
    #     vmin=-1,
    #     vmax=1,
    #     trail_kwargs=trail_kwargs,
    # )

    # projection_kwargs = PROJECTION_KWARGS
    # projection_kwargs["alpha"] = 1
    # projection_kwargs["s"] = 1

    # make_rotating_projection(
    #     pop,
    #     120,
    #     200,
    #     "test_rotation_pdiff.mp4",
    #     cvar=pdiff,
    #     cvar_label="Pressure",
    #     projection_kwargs=projection_kwargs,
    #     n_frames=10,
    #     elev_speed=1,
    #     vmin=-1,
    #     vmax=1,
    #     mode='scatter'
    # )
