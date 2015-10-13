#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Implements state space analysis from:
    Mante, V., Sussillo, D., Shenoy, K. V & Newsome, W. T.
    Context-dependent computation by recurrent dynamics in prefrontal cortex.
    Nature 503, 78–84 (2013).

This implementation relies on a particular representation of your data. It expects
a pandas DataFrame with a Multi-Index that encodes your conditions and time
number, and that units/sensors are encoded in different columns.

For example a data frame like this one should work:
  >>> data = arange(4*4*6).reshape((16,6)) # 3 factors (+time encoding), 4 time points, 6 units
  >>> time = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
  >>> conditions = [[1,2,1,2]*4, [1,1,2,2]*4, [1,1,1,1]*4, time]
  >>> index = pd.MultiIndex.from_arrays(conditions, names=['C1', 'C2', 'C3', 'time'])
  >>> df = pd.DataFrame(data, index=index)

Which looks like this:
                0   1   2   3   4   5
C1 C2 C3 time
1  1  1  1      0   1   2   3   4   5
2  1  1  1      6   7   8   9  10  11
1  2  1  1     12  13  14  15  16  17
2  2  1  1     18  19  20  21  22  23
1  1  1  2     24  25  26  27  28  29
...

You ask why this particular form? It would be best to completely exploit the
structure of the data, i.e. MEG data is trial x time x sensor (x freq) cube and
the dimensions of the cube define an implicit index into the data. Yet, such an
implicit index requires to remember the meaning of each dimension which is a bad
thing when you go back to the data a few weeks from now. Modern data structures
like pandas.DataFrame allow efficient data manipulation while explicitly defining
the semantics of the index (i.e. named indices and data dimensions). The above
structure is a compromise between exploiting structure (units are represented
continuosly) and being able to use pandas. Maybe other projects will allow to
annotate N-dimensional data soon (xray).
'''

import sys, logging
from sklearn import linear_model
from numpy import *
from itertools import product, izip
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.stats import nanmean
from ocupy import datamat
import patsy
from named_array import AxisArray
import pandas as pd

def zscore(data):
    '''
    Z-score a dataset in place.
    '''
    data -= data.mean()
    data /= data.std()


def regression_weights(data, formula, time_label='samplenr'):
    '''
    Estimate linear weights to predict a unit's activity from predictors.

    data is a pandas.DataFrame that encodes sensors as colums and conditions and
    time in it's index (i.e. a multi-index). The index needs to have the levels
    'trial' and 'samplenr', the latter encodes the time of a sample unambigously
    as an int.

    formula is patsy formula description to build a design matrix. The formula
    language is very similar to R's formula language.

    Output:
        betas_nt : pandas.DataFrame
            Encodes for each unit and time point the regression weights for all
            factors occuring in the formula.
    '''
    ntime = data.shape[1]

    def get_weights(unit, unit_data):
        regs = patsy.dmatrix(formula, data=unit_data.reset_index())
        labels = regs.design_info.column_names
        coefs = []
        for time, data  in unit_data.groupby(level=time_label):
            idx = isnan(data.values)
            next_entry = dict((name, nan) for name in labels)
            next_entry[time_label] = time
            next_entry['unit'] = unit
            if not all(idx):
                fit = linear_model.LinearRegression(
                    fit_intercept=False).fit(regs[~idx, :], data.values[~idx])
                for name, value in zip(labels, fit.coef_):
                    next_entry[name] = value
            coefs.append(next_entry)
        df = pd.DataFrame(coefs)
        return df.set_index([time_label, 'unit'])


    dfs = []
    for unit in data:
        dfs.append(get_weights(unit, data[unit]))
    betas_nt = pd.concat(dfs)
    return betas_nt


def predict_unit(unit, formula, bnt):
    '''
    Predict activiy of one unit. TODO: Document this.
    '''
    trial_dict = dict((f, unit.field(f)) for f in unit.fieldnames())
    dmp = patsy.dmatrix(formula, data=trial_dict)
    pred = nan*ones((len(unit), unit.data.shape[1]))
    for i, (u, d) in enumerate(zip(unit.unit, dmp)):
        coefs = squeeze(bnt[bnt.unit == u].coef)
        pred[i, :] = dot(d, coefs.T)
    return pred


def condition_matrix(data, conditions, filter=lambda x:gaussian_filter(x, 15), time_label='samplenr'):
    '''
    Computes a matrix that contains concatenated time series (across conditions)
    for each unit. The size of the matrix is Nunit x (Nsamples * Nconditions)

    Input:
        data : pd.DataFrame
        conditions : list of condition names.
            The condition names in this list map to *levels* in the dataframe.
    Output:
        Matrix : Nunit x (Nsamples * Nconditions)
    '''
    maps = []
    for _, condition in data.groupby(level=list(conditions)):
        condition = condition.groupby(level=time_label).mean()
        if filter is not None:
            condition = condition.apply(filter)
        maps.append(condition)
    return vstack(maps).T


def pca_cleaning(X, N_components=12):
    '''
    Performs PCA based data cleaning on the population responses.

    X is a num_units x (time * conditions) matrix that contains in each row
    the response of a unit. Each response is a unit's time-series in a condition
    and all conditions relevant to an analysis are concatenated.

    The PCA is performed with each column as one observation, the dimensionality
    of the space is Nunit.
    '''
    pca = PCA(n_components=min(X.shape[0], N_components))
    logging.info('Fitting PCA = %i components' % N_components)
    pca.fit(X.T)
    ks = []
    for v in pca.components_:
        v = v[:, newaxis]
        ks.append(dot(v, v.T))
    D = array(ks).sum(0)
    logging.info('Transforming X')
    Xpca = dot(D, X)
    return Xpca, D, pca.explained_variance_ratio_, pca


def regression_embedding(bnt, D, factors):
    '''
    Find maximum norm regresion coefficients over time, project this into
    PCA cleaned space and orthogonalize regression vectors.

    Returns:
        Q : AxisArray (array with named colums)
            Orthogonalized axes in state space.
        Bmax : AxisArray
            Non-orthogonalized axes in state space.
        bmax_list : list
            List of timepoints where Bmax occurs
        norms : list
            Norm of axes vectors, time given by index.
        maps :  list
            Coefficients for each unit and time point.
    '''
    Bmax, bmax_list, norms, maps, = [], [], [], []
    for factor in factors:
        coefs = bnt[factor].unstack().values.T
        b = [dot(D, coefs[:, i]) for i in range(coefs.shape[1])]
        maps.append(array(b))
        norm_b = [linalg.norm(bb) for bb in b]
        bmax = b[argmax(norm_b)]
        Bmax.append(bmax)
        bmax_list.append(where(b == bmax)[0][0])
        norms.append(norm_b)
    Bmax = AxisArray(Bmax, factors).T
    Q, r = linalg.qr(Bmax)
    return Q, Bmax, bmax_list, norms, maps

def embedd(data, formula, conditions=None, N_components=12,
            time_label='samplenr', filter=lambda x:gaussian_filter(x, 15)):
    '''
    Statespace analysis wrapper function. Use this function for your analysis.

    Input:
        data: ocupy.datamat
        formula: Patsy formula that defines the regression model.
    '''
    logging.info('Getting regression weights')
    bnt = regression_weights(data, formula, time_label=time_label)

    if conditions is None:
        conditions = [f for f in bnt.columns]
        # Patsy puts the Intercept into the front, which is usually not what is
        # wanted for the QR decomposition.
        if conditions[0] == 'Intercept':
            conditions = conditions[1:] + [conditions[0]]
        logging.info("Using this condition ordering: " + ' + '.join(conditions))

    logging.info('Building condition matrix')
    X = condition_matrix(data, set(conditions) - set(['Intercept']), time_label=time_label,
                         filter=filter)

    logging.info('Doing PCA cleaning')
    Xpca, D, exp_var, pca = pca_cleaning(X, N_components)

    logging.info('Regression embedding')
    Q, Bmax, t_bmax, norms, maps = regression_embedding(bnt, D, conditions)

    return Q, Bmax, bnt, D, t_bmax, norms, maps, exp_var, pca, X


def cv_trajectory(chunker, formula,
        group_test_by=None,
        N_components=12,
        time_label='samplenr', axis_names=None, filter=lambda x:gaussian_filter(x, 15)):
    '''
    '''
    accs = {}
    Qs = []
    for i,(train, test) in enumerate(chunker):
        Q = embedd(train, 'mc+colorc+1', time_label='time', filter=filter)[0]
        if axis_names is None:
            axis_names = Q.column_names[:2]

        ax1, ax2 = Q.get_axis(axis_names[0]), Q.get_axis(axis_names[1])
        for condition, data in test.groupby(level=group_test_by):

            c_data = condition_matrix(data, group_test_by, time_label='time', filter=filter)
            if condition not in accs:
                accs[condition] = []
            xmc, ycc = get_trj(c_data, ax1, ax2)
            accs[condition].append([xmc, ycc])
        Qs.append(Q)
    return dict((k, array(v)) for k, v in accs.iteritems()), axis_names, Qs



def get_trj(data, axis1, axis2):
    '''
    Project a conditon into the state space.

    data needs to be a num_unit x time matrix, axis1 and axis2 are vectors that
    define axes in state space.
    '''
    return dot(axis1, data), dot(axis2, data)

def get_cov():
    pass
