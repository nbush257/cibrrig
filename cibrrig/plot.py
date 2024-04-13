import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


def plot_laser(intervals,amplitudes,ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    # interleave zeros for the offsets
    new_amps = np.vstack([np.zeros_like(amplitudes),amplitudes]).T.ravel()
    plt.step(intervals.ravel(),new_amps)
    


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
        p = ax.scatter(X[:,dims[0]],X[:,dims[1]],c=c,s=s,alpha=alpha)
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
    