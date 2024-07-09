'''
Standard utility functions
'''
import numpy as np
import pandas as pd
import logging
logging.basicConfig()
_log = logging.getLogger(__name__)
_log.setLevel(level=logging.INFO)

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
    


def parse_opto_log(rr):
    '''
    make a readable string from the opto log
    '''
    label = rr.label.replace('opto_','')
    try:
        np.isnan(rr.phase)
        phase=''
    except:
        phase = rr.phase + ' '
        
    try:
        np.isnan(rr['mode'])
        mode = ''
    except:
        mode= rr['mode']+' '
        
    if 'amplitude_mw' in rr.keys():
        amp = f'{rr.amplitude_mw:2.1f}mW '
    else:
        amp = f'{rr.amplitude:1.2f}V '
    
    if np.isnan(rr.frequency):
        freq = ''
    else:
        freq = f'{rr.frequency:2.0f}Hz '
    
    if np.isnan(rr.pulse_duration):
        pulse_dur = ''
    else:
        pulse_dur = f'{rr.pulse_duration*1000:.0f}ms'
    
    
    out = f'{label} {phase}{mode}{amp}{freq}{pulse_dur}'
    return(out)


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

def time_to_interval(ts,starts,stops=None,labels=None):
    """
    For each event in ts, assign it to an interval defined by starts
    Stops can be provided to close the interval. If stops is not provided
    then the interval is closed by the next event in starts

    Args:
        ts (1D numpy array): Array of event times to classify
        starts (1D numpy array): Start times of all the intervals to define
        stops (1D numpy array, optional): Stop times of all the intervals. Must be the same length as starts. Defaults to None.
        labels (1D numpy array or list, optional): Labels for each interval. Must be the same size as starts. Defaults to None.
    """
    # Sanitize starts and stops
    if stops is not None:
        validate_intervals(starts,stops)
    else:
        stops = np.concatenate([starts[1:],[np.inf]])

    # Preallocate group
    group = np.zeros(ts.shape[0],dtype='int')*np.nan
    for ii,(start,stop) in enumerate(zip(starts,stops)):
        mask = np.logical_and(ts>start,ts<stop)
        group[mask] = ii
    
    # Map integer group to label
    if labels is not None:
        assert(len(labels) == len(starts))
        mapper = {k:v for k,v in enumerate(labels)}
        group = np.vectorize(mapper.get)(group)

    return(group)


def make_pre_post_trial(alf_object,intervals,conditions=None,window=None,pad=0,vars=None,wide=False,agg_func='mean'):
    """
    Gets paired test/control data from a set of intervals (trials)
    Optionally and flexibly accepts combinatorial condition assignments for each interval
    (e.g. grid search on stimulus frequency and amplitude)

    Takes an alf object with a times attribute and finds all the 
    observations of that variable in each interval. Then comptues a 
    "control" period that immediately precedes each test interval.


    Can output in "wide" form by creating a pandas pivot table where rows are trial number, split into columns for test and control


    Example use case: getting the average value of a variable before and during an opto stimulus train

    If conditions is a pandas dataframe with multiple columns, keeps the column name in the output

    Args:
        alf_object (AlfBunch): Alf data to aggregate
        intervals (2D numpy array): N x 2 of start and stop times
        conditions (list,numpy array, or pandas dataframe): Conditions to assign each interval to (e.g., stimulus frequency). Default to None
        window (float, optional): Size of the control window in seconds. If None, uses the same duration as the test duration. Defaults to None.
        pad (int, optional): Seconds before the test interval to exclude. Defaults to 0.
        vars (list,optional): Variables in alf object to aggregate
        wide (bool,optional): If true, makes the data wideform
    """    
    starts = intervals[:,0]
    stops = intervals[:,1]

    # Create control intervals to preceed the test intervals
    window = window or stops-starts
    control_starts = intervals[:,0]-window-pad
    control_stops = starts-pad

    if np.any(control_starts<0):
        _log.warning(f"control starts has entries less than 0:\n{control_starts}")

    # Unpack
    assert('times' in alf_object.keys()),'Input ALF object must have attribute "times"'
    ts = alf_object.times

    use_conditions = False
    if conditions is not None:
        assert(len(conditions)==intervals.shape[0]),f"Length of conditions ({len(conditions)}) must match length of intervals ({intervals.shape[0]})"
        use_conditions = True
        if type(conditions)==list or type(conditions)==np.ndarray:
            conditions = pd.DataFrame(conditions,columns=['condition'])
    else:
        conditions = pd.DataFrame(np.array(['test']*len(starts)), columns=['condition'])
    conditions = conditions.reset_index(drop=True)
    
    categories = conditions.columns.to_list()
    grouped = conditions.groupby(categories)

    # Initialize condition and trial arrays
    condition = pd.DataFrame(np.full([ts.shape[0],len(categories)],np.nan,dtype=object),columns=conditions.columns)
    comparison = np.full_like(ts, np.nan, dtype=object)
    trial = np.full_like(ts, np.nan, dtype=float)

    # for category in categories:
    #     unique_conditions = conditions[category].unique()
    #     for cond in unique_conditions:
    #         # Subset
    for group_keys, group in grouped:
        # this_condition = conditions[category]==cond
        # this_condition = this_condition.values
        _starts = starts[group.index]
        _stops = stops[group.index]
        _control_starts = control_starts[group.index]
        _control_stops = control_stops[group.index]

        # Assign observation indicies to intervals
        test_trial_num = time_to_interval(ts,_starts,_stops)
        control_trial_num = time_to_interval(ts,_control_starts,_control_stops)

        # Make sure observations are not assigned to both test and control
        overlap = np.logical_and(np.isfinite(test_trial_num),np.isfinite(control_trial_num))
        if np.any(overlap):
            _log.warning(f"Observations{np.where(overlap)} are assigned to both test and control")

        # Assign condition and trial numbers for test and control
        condition.loc[np.isfinite(test_trial_num),categories] = group_keys
        condition.loc[np.isfinite(control_trial_num),categories] = group_keys

        comparison[np.isfinite(test_trial_num)] = 'test'
        comparison[np.isfinite(control_trial_num)] = 'control'

        trial[np.isfinite(test_trial_num)] = test_trial_num[np.isfinite(test_trial_num)]
        trial[np.isfinite(control_trial_num)] = control_trial_num[np.isfinite(control_trial_num)]


    # Pandas manipulations to shape the output data
    out_df = alf_object.to_df()
    if vars:
        if type(vars)!=list:
            vars = [vars]
        out_df = out_df[vars]
    if use_conditions:
        out_df = out_df.merge(condition,left_index=True,right_index=True)
    out_df['trial'] = trial
    out_df['comparison'] = comparison
    out_df.dropna(axis=0,subset=['trial'],inplace=True)
    out_df['trial'] = out_df['trial'].astype('int')
    out_df = out_df.reset_index(drop=True)
    var_rename = {x:x+'_'+agg_func for x in vars}
    out_df = out_df.rename(var_rename,axis=1)
    if not use_conditions:
        categories.remove('condition')
    agg_data = out_df.groupby(categories+['comparison','trial']).agg(agg_func).reset_index()

    if wide:
        if use_conditions:
            agg_data = pd.pivot_table(out_df,values=vars,columns=categories+['comparison'],index='trial')
        else:
            agg_data = pd.pivot_table(out_df,values=vars,columns=['comparison'],index='trial')

    return(agg_data)


def get_pct_diff(df,vars,condition_names=None):
    """Given a long form output from "make_pre_post_trial" calculate the percent difference

    Args:
        df (pandas dataframe): long form pandas data. Must have columns['trial','comparison'] Comparison must be either ['test','control']
        condition_names (list): columns to treat as conditions (Categories)
        vars (list): columns to treat as variables and compute %diff on, must be numeric
    """    
    condition_names = condition_names or []
    control_df = df[df['comparison'] == 'control']
    test_df = df[df['comparison'] == 'test']
    merged_df = pd.merge(control_df, test_df, on=condition_names+['trial'], suffixes=('_control', '_test'))

    # Calculate the difference in inst_freq
    for vv in vars:
        x = merged_df[f'{vv}_test']
        y = merged_df[f'{vv}_control']
        merged_df[f'{vv}_pct_diff'] = 100*((x-y)/y)
        merged_df[f'{vv}_diff'] = x-y
    
    out_columns = condition_names+['trial']+[f'{vv}_pct_diff' for vv in vars] + [f'{vv}_diff' for vv in vars]
    result_df = merged_df[out_columns]
    return(result_df)