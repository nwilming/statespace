#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Implement decoding models for MEG/EEG data, well, who cares - it also works for
other spatiotemporal data.

This script expects data to be in a standardized pandas.DataFrame. The layout of
the data frame is:
    - Each colum is one feature, each row is one observation.
    - The index contains a level that contains labels for decoding.

Decoding returns a data frame with a prediction for each test label. This time
labels and predictions

'''
import numpy as np
import pandas as pd
import sklearn

def get_test_data(num_ftrs=20, num_samples=200, num_time=100):
    from sklearn.datasets import samples_generator
    def make_timepoint(t, sep=1):
        centers = [[0]*num_ftrs, [sep]*num_ftrs]
        X, y = samples_generator.make_blobs(num_samples, n_features=num_ftrs,
            centers=centers, cluster_std=0.25)
        df = pd.DataFrame(X, columns=np.arange(0,num_ftrs))
        df['label'] = y
        df['samplenr'] = t
        df.set_index(['label', 'samplenr'], inplace=True)
        return df
    return pd.concat([make_timepoint(t,sep=s) for t, s in enumerate(np.linspace(1,0,num_time))])


def decode(train, test, model, condition, transform=None):
    '''
    Perform one iteration of training and test.
    '''
    train_labels = train.index.get_level_values(condition)
    test_labels = test.index.get_level_values(condition)
    mdl = model.fit(train.values, train_labels)
    prediction = mdl.predict(test.values)
    return prediction, test_labels, mdl


def test_chunker(data):
    yield data, data

def chunker(data, field):
    '''
    Chunk datasets into blocks and then match consecutive
    blocks as test and training set.
    '''
    chunks = list(data.groupby(level=field))
    for idx in range(len(chunks)):
        idmod1 = np.mod(idx+1, len(chunks))
        yield chunks[idx][1], chunks[idmod1][1]

def leave_one_out(data, field):
    '''
    Chunk datasets into blocks and then match consecutive
    blocks as test and training set.
    '''
    values = data.index.get_level_values(field)
    for idx in np.unique(values):
        mask = values==idx
        yield data[mask], data[~mask]

def stratified(data, target, K=3):
    cv = sklearn.cross_validation.StratifiedKFold(
        data.index.get_level_values(target), K)
    for train, test in cv:
        yield data.ix[train,:], data.ix[test, :]

def decode_across_level(train, test, model, condition, level='samplenr'):
    '''
    Split up train and test acording to values in a level and decode.
    This can, for example, be used to decode for each time point.
    '''
    results = []
    models = []
    for train_t, test_t in zip(train.groupby(level=level),
                               test.groupby(level=level)):
        p, l, mdl = decode(train_t[1], test_t[1], model(), condition)
        results.append(pd.DataFrame({level:[train_t[0]]*len(p),
                                     'trial':np.arange(len(p))+1, 'predicted':p,
                                     'label':l}))
        models.append(mdl)
    return pd.concat(results), models

def accuracy(df, level='samplenr'):
    if 'fold' in df.columns:
        results = np.ones((len(np.unique(df[level])), len(np.unique(df.fold))))*np.nan
        for (i,j), val in df.groupby([level, 'fold']):
            results[i,j] = sum(val.label == val.predicted)/float(len(val))
    else:
        results = np.ones((len(np.unique(df[level])), 1))*np.nan
        for i, val in df.groupby([level]):
            results[i,0] = sum(val.label == val.predicted)/float(len(val))
    return results

def do(chunker, model, condition, time_level='samplenr'):
    accs = []
    models = []
    for i,(train, test) in enumerate(chunker):
        acc, mdl = decode_across_level(train, test, model, condition, level=time_level)
        models.append(mdl)
        acc['fold'] = i
        accs.append(acc)
    return pd.concat(accs), models
