import h5py
from numpy import *
import pandas as pd
from itertools import product as iproduct
import glob, os, sys
from pylab import *
import statespace as st

subjects = [12, 13, 15, 16, 17, 10, 9, 8, 2, 3, 6, 7, 4, 5, 20, 21, 14, 19, 18]
subject_files = {}
tfr_files = {}
for sub in subjects:
    sessions =  glob.glob('/Volumes/dump/Data-Anne/P%02i*cleandata.mat'%sub)
    subject_files[sub] = zip(range(len(sessions)), sessions)
    sessions = glob.glob('/home/aurai/Data/MEG-PL/P%02i/MEG/TFR/*_all_freq.mat'%sub)
    tfr_files[sub] = zip(range(len(sessions)), sessions)


def df_from_fieldtrip(data, index_labels=None):
    '''
    Read a fieldtrip data structure into a data frame.
    '''
    collect = []
    index = []
    labels = []
    for label in data['data']['label'][:].T:
        labels.append(''.join([unichr(t) for t in data[label[0]]]).encode('utf8'))
    trialinfo = data['data']['trialinfo'][:,:]
    trial_data = data['data']['trial']
    trial_time = data['data']['time']
    channels = [(i, t) for i, t in zip(range(len(labels)), labels) if t.startswith('M')]
    for j, (td, ti, tt) in enumerate(zip(trial_data, trialinfo.T, trial_time)):
        td_vals = data[td[0]] # this is the data for this trial
        tt_vals = data[tt[0]] # this is the time for this trial
        ind = tile(ti, (len(tt_vals), 1))
        ind = hstack((tt_vals[:]*0+j, ind, tt_vals))
        index.append(ind)
        collect.append(td_vals)
    index = vstack(index)
    collect = vstack(collect)
    ind = pd.MultiIndex.from_arrays(index.T, names=index_labels)
    ind_col = pd.MultiIndex.from_arrays([labels], names='Sensors')
    return pd.DataFrame(collect, index=ind, columns=ind_col)

def clean_anne_broadband(df):
    '''
    Clean up data frames from Anne's data.
    '''
    df.index = df.index.droplevel(['trial onset', 'ref onset',
                                 'interval onset', 'stim onset',
                                 'feedback onset', 'feedback', 'response hand', 'response sample'])
    return df

def select(trial, start='trial onset', end=None, offset=0.0, baseline_start='trial onset', baseline_end='ref onset', baseline_offset=0.1):
    '''
    Select an epoch from a single trial, possibly with baseline correction.
    '''
    gtl =  lambda x: trial.index.get_level_values('time')[int(trial.index.get_level_values(x)[0])]
    start_time = gtl(start)
    end_time = start_time

    if end is not None:
        end_time = gtl(end)
    end_time += offset
    time = trial.index.get_level_values('time')

    if baseline_start is not None and baseline_end is not None:
        baseline_start = gtl(baseline_start) + baseline_offset
        baseline_end = gtl(baseline_end) - baseline_offset
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

def subject(subject, files, selector):
    dfs = []
    for session_id, f in files:
        session = h5py.File(f)
        df = df_from_fieldtrip(session,
            index_labels=['trial_index', 'trial onset', 'ref onset', 'interval onset', 'stimulus',
                          'stim onset', 'response hand', 'choice', 'correct',
                          'response sample', 'feedback', 'feedback onset', 'trial',
                          'block', 'session', 'time'])
        df = epoch(df, selector)
        df.sort_index(inplace=True)
        df = clean_anne_broadband(df)
        # Add subject and session levels
        df['session_num'] = session_id
        df['subject'] = subject
        df.set_index('session_num', append=True, inplace=True)
        df.set_index('subject', append=True, inplace=True)
        dfs.append(df)
    return pd.concat(dfs)

default_selector = lambda x: select(x, start='stim onset', offset=0.75)
