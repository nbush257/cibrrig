import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from .utils.utils import weighted_histogram,parse_opto_log
from one.alf.io import AlfBunch
import seaborn as sns

laser_colors ={
473:'#00b7ff',
565:'#d2ff00',
653:'#ff0000'
}

def plot_laser(laser_in,**kwargs):
    '''
    kwargs:
    amplitudes
    ax
    mode (shade,bar,vline)
    amp_label
    wavelength
    alpha
    **plot kwargs
    '''
    if isinstance(laser_in,AlfBunch):
        if 'category' in laser_in.keys():
            _plot_laser_log(laser_in,**kwargs)
        else:
            _plot_laser_alf(laser_in,**kwargs)
    else:
        _plot_laser_intervals(laser_in,**kwargs)

def _plot_laser_alf(laser_in,**kwargs):
    intervals = laser_in.intervals
    if 'amplitudesMilliwatts' in laser_in.keys():
        amplitudes = laser_in.amplitudesMilliwatts
        amp_label='mW'
    elif 'amplitudesVolts' in laser_in.keys():
        amplitudes = laser_in.amplitudesVolts
        amp_label='command volts'
    else:
        amplitudes = None


    _plot_laser_intervals(intervals,amplitudes,amp_label=amp_label,**kwargs)


def _plot_laser_intervals(intervals,amplitudes=None,ax=None,mode='shade',amp_label='',wavelength=473,alpha=0.2,**kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    else:
        ax = ax.twinx()

    try:
        iter(alpha)
        alpha_list = True
    except:
        alpha_list = False

    if mode=='shade':
        for ii,stim in enumerate(intervals):
            aa = alpha[ii] if alpha_list else alpha
            ax.axvspan(stim[0],stim[1],color = laser_colors[wavelength],alpha = aa,**kwargs)
    elif mode=='bar':
        yy = ax.get_ylim()[1]
        yy = np.ones_like(intervals[:,0])*yy
        ax.hlines(yy,intervals[:,0],intervals[:,1],color = laser_colors[wavelength],**kwargs)
    elif mode=='vline':
        y0 = ax.get_ylim()[0]
        y0 = np.ones_like(intervals[:,0])*y0

        y1 = ax.get_ylim()[1]
        y1 = np.ones_like(intervals[:,0])*y1
        ax.vlines(intervals[:,0],y0,y1,color=laser_colors[wavelength],**kwargs)
    else:
    # interleave zeros for the offsets
        if amplitudes is None:
            new_amps = np.vstack([np.zeros_like(intervals[:,0]),np.ones_like(intervals[:,0])]).T.ravel()
        else:
            new_amps = np.vstack([np.zeros_like(amplitudes),amplitudes]).T.ravel()
        ax.step(intervals.ravel(),new_amps,color=laser_colors[wavelength],**kwargs)
        ax.set_ylabel(amp_label)
    plt.xlabel('Time (s)')
        

def _plot_laser_log(log,query=None,rotation=45,fontsize=6,**kwargs):
    opto_df = log.to_df().query('category=="opto"')
    intervals = opto_df[['start_time','end_time']].values
    if 'amplitude_mw' in  opto_df.keys():
        amps = opto_df['amplitude_mw']
        amp_units='mW'
    else:
        amps = opto_df['amplitude']
        amp_units='command_volts'
    
    _plot_laser_intervals(intervals,amplitudes=amps,amp_label=amp_units,**kwargs)
    if query:
        opto_df = opto_df.query(query)
    for _, rr in opto_df.iterrows():
        s = parse_opto_log(rr)
        plt.text(np.mean([rr.start_time,rr.end_time]),plt.gca().get_ylim()[1],s,rotation=rotation,fontsize=fontsize)


# TODO: This works, but needs to be expanded to work for 3D and to be more complete.
def plot_projection_line(X,cvar,dims=[0,1],cmap='viridis'):
    # Lines are way slower than scatters
    samps = np.arange(X.shape[0])
    cmap = plt.get_cmap(cmap)  # You can use any other colormap as well

    norm = mcolors.Normalize(vmin=np.min(cvar), vmax=np.max(cvar))

    for s0 in samps:
        plt.plot(X[s0:s0+2,dims[0]],X[s0:s0+2,dims[1]],color=cmap(norm(cvar[s0])))


def plot_projection(X,dims,**kwargs):
    if len(dims)==2:
        return(plot_2D_projection(X,dims,**kwargs))
    elif len(dims)==3:
        return(plot_3D_projection(X,dims,**kwargs))
    else:
        raise(ValueError(f'Number of plotted dimensions must be 2 or 3. {dims=}'))
    

def plot_3D_projection(X,dims=[0,1,2],cvar=None,ax=None,title='',
                       s = 1,
                       vmin=None,vmax=None,cmap='viridis',c='k',alpha=0.2,
                       lims = [-4,4],plot_colorbar=True,colorbar_title='',
                       pane_color=None):
    assert(len(dims)==3), f"Must choose 3 dimensions to plot. Chose {dims}"
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111,projection='3d')
    else:
        f = ax.get_figure()

    assert isinstance(ax,Axes3D),'ax must be a 3D projection'


    if cvar is None:
        p = ax.scatter(X[:,dims[0]],X[:,dims[1]],X[:,dims[2]],c=c,s=s,alpha=alpha)
    else:
        vmin = vmin or np.min(cvar)
        vmax = vmax or np.max(cvar)
        p = ax.scatter(X[:,dims[0]],X[:,dims[1]],X[:,dims[2]],c=cvar,s=s,cmap=cmap,alpha=alpha,
                       vmin=vmin,vmax=vmax)
        if plot_colorbar:
            cax = f.add_axes([0.25, 0.85, 0.5, 0.02])
            cbar = f.colorbar(p,cax=cax,orientation='horizontal')
            cbar.set_label(colorbar_title)
            cbar.solids.set(alpha=1)
    
    ax.set_title(title)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_zlim(lims)

    ax.set_xlabel(f'Dim {dims[0]+1}')
    ax.set_ylabel(f'Dim {dims[1]+1}')
    ax.set_zlabel(f'Dim {dims[2]+1}')

    ax.grid(False)

    if pane_color is not None:
        ax.xaxis.set_pane_color(pane_color)  # Set the color of the x-axis pane
        ax.yaxis.set_pane_color(pane_color)  # Set the color of the y-axis pane
        ax.zaxis.set_pane_color(pane_color)  # Set the color of the z-axis pane

    return(f,ax)
    

def plot_2D_projection(X,dims=[0,1],cvar=None,ax=None,title='',
                       s=1,vmin=None,vmax=None,cmap='viridis',c='k',alpha=0.2,
                       lims=[-4,4],plot_colorbar=True,colorbar_title=''):

    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    else:
        f = ax.get_figure()
    assert(len(dims)==2), f'Must choose 2 dimensions to plot. Chose {dims}'


    if cvar is None:
        p = ax.scatter(X[:,dims[0]],X[:,dims[1]],color=c,s=s,alpha=alpha)
    else:
        vmin = vmin or np.min(cvar)
        vmax = vmax or np.max(cvar)
        p = ax.scatter(X[:,dims[0]],X[:,dims[1]],c=cvar,s=s,cmap=cmap,alpha=alpha,
                       vmin=vmin,vmax=vmax)
        if plot_colorbar:
            cax = f.add_axes([0.25, 0.85, 0.5, 0.02])
            cbar = f.colorbar(p,cax=cax,orientation='horizontal')
            cbar.set_label(colorbar_title)
            cbar.solids.set(alpha=1)
    
    ax.set_title(title)

    ax.axis('square')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f'Dim {dims[0]+1}')
    ax.set_ylabel(f'Dim {dims[1]+1}')
    ax.spines[['right', 'top']].set_visible(False)


    return(f,ax)
    

def clean_polar_axis(ax):
    ax.set_yticks([ax.get_yticks()[-1]])
    ax.set_xticks([0,np.pi/2,np.pi,np.pi*3/2])
    ax.set_xticklabels(['0','$\\frac{\pi}{2}$','$\pi$','$\\frac{-\pi}{2}$'])


def clean_linear_radial_axis(ax):
    ax.set_xticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi])
    ax.set_xticklabels(['$-\pi$','$\\frac{-\pi}{2}$','0','$\\frac{\pi}{2}$','$\pi$'])
    sns.despine(trim=True)


def plot_polar_average(x,y,t,ax=None,t0=None,tf=None,color='k',bins=50,multi='sem',alpha=0.3,**plot_kwargs):
    #TODO: Sanitize t0,tf to be iterable
    #TODO: Sanitze

    try:
        iter(t0)
    except:
        t0 = [t0]

    try:
        iter(tf)
    except:
        tf = [tf]

    try:
        iter(color)
    except:
        color = [color]
    assert(len(t0)==len(tf)),f'{len(t0)=} and {len(tf)=}; they must have same shape'

    
    n = len(t0)
    y_polar_out = []

    for ii,(start,stop) in enumerate(zip(t0,tf)):
        s0,sf = np.searchsorted(t,[start,stop])
        phase_bins,y_polar = weighted_histogram(x[s0:sf],y[s0:sf],bins=bins,wrap=True)
        y_polar_out.append(y_polar)

    y_polar_out = np.vstack(y_polar_out)
    m = np.mean(y_polar_out,0)

    # Plotting
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(projection='polar')
    else:
        f = None

    if multi=='sem':
        lb = m - np.nanstd(y_polar_out,0)/np.sqrt(y_polar_out.shape[0])
        ub = m + np.nanstd(y_polar_out,0)/np.sqrt(y_polar_out.shape[0])
        ax.plot(phase_bins,np.mean(y_polar_out,0),color=color[0],**plot_kwargs)
        ax.fill_between(phase_bins,lb,ub,color=color[0],alpha=alpha)
    elif multi=='std':
        lb = m - np.nanstd(y_polar_out,0)
        ub = m + np.nanstd(y_polar_out,0)
        ax.plot(phase_bins,np.mean(y_polar_out,0),color=color[0],**plot_kwargs)
        ax.fill_between(phase_bins,lb,ub,color=color[0],alpha=alpha)
    else:

        for ii,y_polar in enumerate(y_polar_out):
            if len(color)==1:
                c = color[0]
            else:
                c = color[ii]
            ax.plot(phase_bins,y_polar,color=c,**plot_kwargs)
    clean_polar_axis(ax)
    return(f,ax,y_polar_out,phase_bins)
