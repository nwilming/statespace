'''
TODOs:
    - Broadbend activity
    - topological plots of regression weights
    - baseline correction (single trial vs. average)
    - cross validation
'''
import h5py
from numpy import *
import pandas as pd
from itertools import product as iproduct
import glob, os, sys
from pylab import *
import statespace as st


def _trial_to_df(trial_index, data, td, ti, tt, labels, index_labels=None):
    td_vals = data[td[0]] # this is the data for this trial
    tt_vals = data[tt[0]] # this is the time for this trial
    ind = tile(ti, (len(tt_vals), 1))
    ind = hstack((tt_vals[:]*0+trial_index, ind, tt_vals))
    ind = pd.MultiIndex.from_arrays(ind.T, names=index_labels)
    ind_col = pd.MultiIndex.from_arrays([labels], names='Sensors')
    return pd.DataFrame(td_vals[:,:], index=ind, columns=ind_col)


def df_from_fieldtrip(data, index_labels=None, select=lambda x:x):
    '''
    Read a fieldtrip data structure into a data frame.
    '''
    collect = []
    labels = []
    for label in data['data']['label'][:].T:
        labels.append(''.join([unichr(t) for t in data[label[0]]]).encode('utf8'))
    trialinfo = data['data']['trialinfo'][:,:]
    trial_data = data['data']['trial']
    trial_time = data['data']['time']
    channels = [(i, t) for i, t in zip(range(len(labels)), labels) if t.startswith('M')]
    for trial_index, (td, ti, tt) in enumerate(zip(trial_data, trialinfo.T, trial_time)):
        collect.append(
            select(_trial_to_df(trial_index, data, td, ti, tt, labels, index_labels)))
    return pd.concat(collect)



def select(trial, start='trial onset', end=None, offset=(0, 0),
           baseline_start='trial onset', baseline_end='ref onset',
           baseline_offset=(0, 0)):
    '''
    Select an epoch from a single trial, possibly with baseline correction.
    '''
    gtl =  lambda x: trial.index.get_level_values('time')[int(trial.index.get_level_values(x)[0])]
    # Start point selection:
    assert (start is not None) and (end is not None)
    start_time = gtl(start) + offset[0]
    end_time = gtl(end) + offset[1]

    time = trial.index.get_level_values('time')

    if (baseline_start is not None) and (baseline_end is not None):
        baseline_start = gtl(baseline_start) + baseline_offset[0]
        baseline_end = gtl(baseline_end) + baseline_offset[1]
        mask_baseline = (baseline_start < time) & (time < baseline_end)
        baseline = trial.loc[mask_baseline,:].mean()
        trial -= baseline

    mask_trial = (start_time <= time) & (time <= end_time)
    trial = trial.loc[mask_trial,:]
    trial.loc[:, 'samplenr'] = arange(len(trial))
    trial.set_index('samplenr', append=True, inplace=True)
    # Adjust time index: Seems like an unnecessarily complex way of doing it.
    ordering = list(set(trial.index.names) - set(['time'])) + ['time']
    index = trial.index.reorder_levels(ordering)
    new_time = linspace(0, end_time-start_time, len(trial))
    for i in xrange(len(index.values)):
        t = list(index.values[i][:-1]) + [new_time[i]]
        index.values[i] = tuple(t)
    trial.index = pd.MultiIndex.from_tuples(index.values, names=index.names)
    return trial


def epoch(df, func):
    dfs = []
    for _, d in df.groupby(level='trial_index'):
        dfs.append(func(d))
    return pd.concat(dfs)

def downsample_with_averaging(df, window_size, nth=10):
    d = []
    for a,b in df.groupby(level=['session_num', 'trial_index']):
        d.append( pd.rolling_mean(b, window_size).loc[::nth,:])
    return pd.concat(d)
