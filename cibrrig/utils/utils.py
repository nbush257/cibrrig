'''
Standard utility functions
'''
import numpy as np

# May be better in a utils file
def validate_intervals(starts,stops):
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


def event_counts_per_epochs(event_times,starts,stops,rate=False):
    """
    Count the number of occurances of an event in a set of intervals


    Args:
        event_times (1D numpy array): timestamps of each event in seconds
        starts (_type_): start time of each epoch in seconds
        stops (_type_): end time of each epoch in seconds
        rate (bool, optional): If true, divides the count by the duration to get the event rate. Defaults to False.
    """    
    validate_intervals(starts,stops)
    count = np.zeros(len(starts),dtype='int64')
    for ii,(start,stop) in enumerate(zip(starts,stops)):
        idx = np.logical_and(event_times>start,event_times<stop)
        count[ii] = np.sum(idx)

    if rate:
        durs = stops-starts
        return(count/durs)
    else:
        return(count)
    



def weighted_histogram(x,weights,bins,wrap=False):
    """
    Compute the weighted histogram of a variable x
    e.g.: compute PC1 as a function of respiratory phase (phi).:
    weighted_histogram(phi,PC1,np.linspace(-np.pi,np.pi),wrap=True)

    
    Args:
        x (_type_): Variable on the x-axis
        weights (_type_): variable on the y axis
        bins (int or array): either the number of bins on x, or explicit bin edges.
        wrap (bool, optional): whether to duplicate the first bin and add it to the end. Good for circular variables. Defaults to False.
    """    
    assert(x.size == weights.size)
    prior,bins = np.histogram(x,bins)
    likli=[]
    post,bins = np.histogram(x,weights=weights,bins=bins)
    likli = post/prior
    bins = bins[1:]-np.mean(np.diff(bins))
    if wrap:
        bins = np.concatenate([bins, [bins[0]]])
        likli = np.concatenate([likli, [likli[0]]])
    return(bins,likli)