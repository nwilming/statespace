#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Analysis scripts for Anne's data.
'''

import h5py
from numpy import *
import pandas as pd
from itertools import product as iproduct
import glob, os, sys
from pylab import *
import statespace as st
import analysis
import sklearn
from sklearn.lda import LDA
from sklearn.qda import QDA
import decoding as dcd


response_selector = lambda x: analysis.select(x, start='response sample', end='response sample', offset=(-0.5, 0),
                                           baseline_start='trial onset', baseline_end='ref onset',
                                           baseline_offset=(0.1, -0.1))
first_stim = lambda x: analysis.select(x, start='ref onset', end='ref onset', offset=(0, 0.75),
                                           baseline_start='trial onset', baseline_end='ref onset',
                                           baseline_offset=(0.1, -0.1))
second_stim = lambda x: analysis.select(x, start='stim onset', end='stim onset', offset=(0, 0.75),
                                           baseline_start='trial onset', baseline_end='ref onset',
                                           baseline_offset=(0.1, -0.1))
selectors = {'response':response_selector, 'first':first_stim, 'second':second_stim}

classifier = {'SVM':sklearn.svm.SVC,
            #'logistic': lambda: sklearn.linear_model.LogisticRegressionCV(cv=10, dual=False, class_weight='auto', n_jobs=-1),
            'lda': lambda: LDA(),
            'qda': lambda: QDA()}

def parse_data(location='/Volumes/dump/Data-Anne/P%02i*cleandata.mat', subjects=[2]):
    subject_files = {}
    for sub in subjects:
        sessions =  glob.glob(location%sub)
        subject_files[sub] = zip(range(len(sessions)), sessions)
    return subject_files

rawdata = parse_data()

def clean_anne_broadband(df):
    '''
    Clean up data frames from Anne's data.
    '''
    df.index = df.index.droplevel(['trial onset', 'ref onset',
                                 'interval onset', 'stim onset',
                                 'feedback onset', 'feedback', 'response hand', 'response sample'])
    return df


def process_subject(subject, files, filter):
    '''
    Load a fieldtrip for a subject, parse it into a dataframe, epoch it and clean up
    unnecessary fields.

    Input:
        subject: Subject short code (e.g. a number or what not)
        files: A list of data files for this subject.
            The list has the format [(id, filename), ...]. The id is used to
            identify sessions in the resulting dataframe.
        filter: Function
            A function that selects a subset of data for each trial. It accepts
            a single dataframe that contains data for one trial and returns
            a single dataframe.
    '''
    dfs = []
    trial_offset = 0
    for session_id, f in files:
        session = h5py.File(f)
        df = analysis.df_from_fieldtrip(session,
            index_labels=['trial_index', 'trial onset', 'ref onset', 'interval onset', 'stimulus',
                          'stim onset', 'response hand', 'choice', 'correct',
                          'response sample', 'feedback', 'feedback onset', 'trial',
                          'block', 'session', 'time'], select=filter)
        df.sort_index(inplace=True)
        # Add subject and session levels
        df['session_num'] = session_id
        df['subject'] = subject
        df['unique_trial'] = df.index.get_level_values('trial_index')+trial_offset
        trial_offset = df['unique_trial'].max()
        df.set_index('session_num', append=True, inplace=True)
        df.set_index('subject', append=True, inplace=True)
        df.set_index('unique_trial', append=True, inplace=True)
        dfs.append(df)
    return pd.concat(dfs)

def save_subject(save_dir, subject):
    files = rawdata[subject]
    df = process_subject(subject, files)
    save_file = os.path.join(save_dir, 'P%02i_data.hdf'%subject)
    df.to_hdf(save_file, 'raw')
    del df

def estimate_state_space(cross_validator, selector):
    pass


def average_trials_by_label(data, level, N, foreach=['samplenr', 'session_num']):
    '''
    Average trials of a certain condition to trade off noise vs. sample size.
    '''
    #return pd.concat([df.groupby(lambda x: mod(x, N), level='trial_index').mean()
    #            for a, df in data.groupby(level=foreach+[level], as_index=True)])
    if isinstance(level, basestring):
        level = [level]
    levels = foreach + level
    res = []
    for a, df in data.groupby(level=levels):
        idx = df.index.get_level_values('trial_index')
        s = array([0] + cumsum(diff(idx)!=0).tolist())  //N
        for trial, val in  df.groupby(s):
            #import pdb; pdb.set_trace()
            #res.append(val.mean())
            #print df
            val=val.mean().to_frame().T
            val['trial_index'] = trial
            for name, value in zip(levels, a):
                val[name] = value
            val.set_index(levels+['trial_index'], inplace=True)
            res.append(val)
    return pd.concat(res)


def droplevels(df):
    '''
    Remove unnecessary things for now.
    '''
    df.index = df.index.droplevel([u'stim onset', u'feedback', u'feedback onset',
        u'ref onset', u'trial onset', u'session', u'block', u'interval onset',
        u'response sample', u'response hand', u'time'])
    return df


def decoding(filename, epochs, classifier={'SVM':sklearn.svm.SVC}):
    '''
    Perform decoding.
    '''
    results = pd.DataFrame()
    import sys
    for epoch in epochs:
        df = pd.read_hdf(filename, epoch)
        for target_field in ['choice', 'stimulus']:
            for name, clsf in classifier.iteritems():
                print epoch, target_field, name
                sys.stdout.flush()
                chunker = dcd.leave_one_out(df, 'session_num')
                accuracy = dcd.do(chunker, clsf, target_field)
                accuracy['eppoch'] = epoch
                accuracy['target_field'] = target_field
                accuracy['classifier'] = name
                results = pd.concat([results, accuracy])
                results.to_hdf('/Volumes/dump/Data-Anne/minified/P02_data.hdf', 'decoding')
    return results


def plot_accuracy(data):
    epoch_times = {'first':linspace(0, 0.75, 451), 'second':linspace(1, 1.75, 451),
        'response':linspace(2, 2.5, 301)}
    colors = {'choice':'r', 'stimulus':'b'}
    for i, (c, classifier) in enumerate(data.groupby(['classifier'])):
        subplot(3,1,i+1)
        title(c)
        for (cond, target), d in classifier.groupby(['eppoch', 'target_field']):
            acc = dcd.accuracy(d)
            plot(epoch_times[cond], acc.mean(1), colors[target], label=target)
        ylim([0, 1.])
    legend()

def filter_and_process_subs(save_dir, subject):
    save_file = os.path.join(save_dir, 'P%02i_data.hdf'%subject)
    for name, filter in selectors.iteritems():
        df = process_subject(subject, rawdata[subject], filter)
        df.to_hdf(save_file, name)
        del df
        print 'Finished %s'%name
