import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import logging

logging.basicConfig()
_log = logging.getLogger()
_log.setLevel(logging.INFO)

#TODO: be able to modify only parts of this
PROJECTION_KWARGS = dict(lw=0.5, alpha=0.2, color="C0")
TRAIL_KWARGS = dict(lw=3, alpha=1, color="C1")
HISTORY_KWARGS = dict(color="C1", lw=0.75, alpha=0.5)
AUX_KWARGS = dict(lw=0.5, color="C1")

# TODO: Organize default arguments?
# TODO: plot only low_D
# TODO: plot without stims (Maybe intervals can be empty)


# TODO: allow for 3D
# TODO: allow for rotation
# TODO: Make stim history stay as the stim color
def make_aux_raster_projection_with_stims(
    pop,
    intervals,
    aux,
    aux_t,
    fn_out,
    stim_color,
    aux_label = '',
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
    style='dark_background',
    projection_kwargs=PROJECTION_KWARGS,
    trail_kwargs=TRAIL_KWARGS,
    history_kwargs=HISTORY_KWARGS,
    aux_kwargs = AUX_KWARGS
):
    plt.style.use(style)
    assert lead_in < duration, f"{lead_in=} must be shorter than {duration=}"
    pop.compute_projection()

    t0 = intervals[0, 0] - lead_in
    tf = t0 + duration

    # Set up figure and axes layout =====
    f = plt.figure(figsize=(3, 8), dpi=dpi)
    gs = f.add_gridspec(nrows=15, ncols=1)
    if len(dims)==3:
        ax = f.add_subplot(gs[8:, :],projection='3d')
    else:
        ax = f.add_subplot(gs[8:, :])

    ax_raster = f.add_subplot(gs[1:8, :])
    ax_aux = f.add_subplot(gs[0, :], sharex=ax_raster)

    # Plot baseline
    ax = pop.plot_projection_line(
        dims=dims, t0=t0 - baseline, tf=t0, ax=ax, **projection_kwargs
    )
    # Plot trail================================
    s0, sf = np.searchsorted(pop.tbins, [t0 - trail_length, t0])
    # Plot thick line (current timepoints)
    if len(dims)==3:
        (trail1,) = ax.plot(
            pop.projection[s0:sf, dims[0]], pop.projection[s0:sf, dims[1]],pop.projection[s0:sf,dims[2]], **trail_kwargs
        )  # NB: the comma after trail1 is important.
    else:
        (trail1,) = ax.plot(
            pop.projection[s0:sf, dims[0]], pop.projection[s0:sf, dims[1]], **trail_kwargs
        )  # NB: the comma after trail1 is important.

    # Plot thin line (all previous timepoints)
    s0, sf = np.searchsorted(pop.tbins, [t0 - trail_length, t0])
    if len(dims)==3:
        (history,) = ax.plot(
            pop.projection[s0:sf, dims[0]], pop.projection[s0:sf, dims[1]], pop.projection[s0:sf,dims[2]],**history_kwargs
        )
    else:
        (history,) = ax.plot(
            pop.projection[s0:sf, dims[0]], pop.projection[s0:sf, dims[1]], **history_kwargs
        )

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
        yy[1] / 2,
        aux_label,
        rotation=90,
        ha="right",
        va="center",
        color=aux_kwargs['color'],
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
    s0, sf = np.searchsorted(pop.tbins, [t0, tf])
    xlim = (
        np.min(pop.projection[s0:sf, dims[0]]),
        np.max(pop.projection[s0:sf, dims[0]]),
    )
    ylim = (
        np.min(pop.projection[s0:sf, dims[1]]),
        np.max(pop.projection[s0:sf, dims[1]]),
    )
    if len(dims)==2:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticks([xticks[0], 0, xticks[-1]])
        ax.set_yticks([yticks[0], 0, yticks[-1]])
        plt.tight_layout()
    elif len(dims)==3:
        zlim = (
            np.min(pop.projection[s0:sf, dims[2]]),
            np.max(pop.projection[s0:sf, dims[2]]),
        )
        view_min = np.floor(np.min([xlim[0],ylim[0],zlim[0]]))
        view_max = np.ceil(np.max([xlim[1],ylim[1],zlim[1]]))
        ax.set_xlim([view_min,view_max])
        ax.set_ylim([view_min,view_max])
        ax.set_zlim([view_min,view_max])
        ax.set_xticks([view_min, 0, view_max])
        ax.set_yticks([view_min, 0, view_max])
        ax.set_zticks([view_min, 0, view_max])
        ax.get_proj()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


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
        trail1.set_data(
            pop.projection[s0:sf, dims[0]], pop.projection[s0:sf, dims[1]]
        )  
        if len(dims)==3:
            trail1.set_3d_properties(pop.projection[s0:sf,dims[2]])

        # update the data
        if is_stim:
            trail1.set_color(stim_color)
        else:
            trail1.set_color(trail_kwargs['color'])  # Sets the color

        # Update the histroy trajectory (all previous time points)
        s0, sf = np.searchsorted(pop.tbins, [t0, this_t])
        history.set_data(pop.projection[s0:sf, dims[0]], pop.projection[s0:sf, dims[1]])
        if len(dims)==3:
            history.set_3d_properties(pop.projection[s0:sf,dims[2]])


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
        if len(dims)==3 and  rotate:
            if frames>rotation_delay:
                ax.view_init(ax.elev+elev_speed,ax.azim-azim_speed)  


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


if __name__ == "__main__":
    # Testing:
    _log.info("Testing")
    from one.api import One
    from brainbox.io.one import SpikeSortingLoader
    from cibrrig.analysis.population import Population, get_good_spikes

    CACHE_DIR = "/data/hps/assoc/private/medullary/data/alf_data_repo"
    one = One(CACHE_DIR)
    eid = one.search(subject="m2024-40")[0]
    ssl = SpikeSortingLoader(one, eid=eid)
    spikes, clusters, channels = ssl.load_spike_sorting(spike_sorter="")

    spikes, cluster_ids = get_good_spikes(spikes, clusters)
    pop = Population(spikes.times, spikes.clusters)
    physiology = one.load_object(eid, "physiology")
    dia = one.load_object(eid, "diaphragm")

    intervals = np.array([[10.0, 11.0]])
    make_aux_raster_projection_with_stims(
        pop,
        intervals,
        dia.filtered,
        dia.times,
        "test.mp4",
        aux_label="dia",
        stim_color="c",
        lead_in=1,
        duration=2,
        dpi=100,
        dims=[0,1,2],
        elev_speed=0.4,
        azim_speed=0.4,
    )
