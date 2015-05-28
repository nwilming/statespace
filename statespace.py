'''
Implement state space analysis from Mante et al. 2013 Nature something


'''
import sys
from sklearn import linear_model
from numpy import *
from itertools import product, izip
from sklearn.decomposition import PCA
import pylab as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import nanmean
import matplotlib
from ocupy import datamat
import patsy


def zscore(data):
    for n in unique(data.unit):
        mu, sigma = nanmean(data.data[data.unit==n]), nanstd(data.data[data.unit==n])
        data.data[data.unit==n] = (data.data[data.unit==n]-mu)/sigma


def patsy_regression_weights(data, formula):
    '''
    Regress factors onto predicted variable from data.
    '''
    nunits, ntime = len(unique(data.unit)), data.data.shape[1]
    dm = datamat.AccumulatorFactory()
    for n in unique(data.unit):
        sys.stdout.flush()
        unit = data[data.unit == n]
        datadict = dict((f, unit.field(f)) for f in unit.fieldnames())
        regs = patsy.dmatrix(formula, data=datadict)
        coefs = []
        for t in range(ntime):
            r_tu = unit.data[:, t]
            idx = isnan(r_tu)
            if not all(idx):
                fit = linear_model.LinearRegression(
                    fit_intercept=False).fit(regs[~idx,:], r_tu[~idx])
                coefs.append(fit.coef_)
            else:
                coefs.append(nan*ones((regs.shape[1],)))
        dm.update({'unit': n, 'coef': squeeze(array(coefs))})
    betas_nt = dm.get_dm()
    return betas_nt, regs.design_info.column_names


def predict_unit(unit, formula, bnt):
    trial_dict = dict((f, unit.field(f)) for f in unit.fieldnames())
    dmp = patsy.dmatrix(formula, data=trial_dict)
    pred = nan*ones((len(unit), unit.data.shape[1]))
    for i, (u, d) in enumerate(zip(unit.unit, dmp)):
        coefs = squeeze(bnt[bnt.unit == u].coef)
        pred[i, :] = dot(d, coefs.T)
    return pred


def conmean(dm, **kwargs):
    '''
    Computes population average response for condition
    specified in kwargs.
    '''
    for key, value in kwargs.iteritems():
        dm = dm[dm.field(key) == value]
    return gaussian_filter(nanmean(dm.data, 0), 15)
    

def dict_product(dicts):
    '''
    Cartesian products of values in dicts. Returns one dictionary per
    combination.
    '''
    for key, value in dicts.iteritems():
        try:
            value[0]
        except (IndexError, TypeError):
            dicts[key] = [value]
    return (dict(izip(dicts, x)) for x in product(*dicts.itervalues()))


def condition_matrix(data, conditions):
    '''
    Computes data matrix X which contains one row per unit and colums
    contain the concatenated condition averages for this unit.
    '''
    X = []
    for n in unique(data.unit):
        uv = []
        unit = data[data.unit == n]
        for condition in conditions:
            p = conmean(unit, unit=[n], **condition)
            uv.append(p)
        X.append(concatenate(uv))
    return array(X)


def check_factors(data, factors):
    for condition in dict_product(factors):
        x = conmean(data, **condition)
        if any(isnan(x)):
            print condition, sum(isnan(x))


def pca_cleaning(data, factors):
    '''
    Performs PCA based data cleaning on the population responses.
    
    Important: factors is now a list of valid condition dictionaries
    '''
    X = condition_matrix(data, factors)
    pca = PCA(n_components=min(X.shape[0], 12))
    print 'Fitting PCA'
    sys.stdout.flush()
    pca.fit(X.T)
    ks = []
    for v in pca.components_:
        v = v[:, newaxis]
        ks.append(dot(v, v.T))
    D = array(ks).sum(0)
    print 'Transforming X'
    sys.stdout.flush()
    Xpca = dot(D, X)
    return X, Xpca, D


def regression_embedding(bnt, D):
    '''
    Find maximum norm regresion coefficients over time, project this into 
    PCA cleaned space and orthogonalize regression vectors.
    '''
    Bmax = []
    bmax_list= []
    norms = []
    maps = []
    for v in range(bnt.coef.shape[2]):
        b = bnt.coef[:, :, v]
        b = [dot(D, b[:,i]) for i in range(b.shape[1])]
        maps.append(array(b))
        norm_b = [linalg.norm(bb) for bb in b]
        bmax = b[argmax(norm_b)]
        Bmax.append(bmax)
        bmax_list.append(where(b == bmax)[0][0])
        norms.append(norm_b)
    Bmax = array(Bmax).T
    Q, r = linalg.qr(Bmax)
    return Q, Bmax, bmax_list, norms, maps


def embedd(data, formula, valid_conditions):
    '''
    Statespace analysis wrapper function. Use this function for your analysis.

    Input:
    data: ocupy.datamat

    '''
    print 'Getting regression weights'
    sys.stdout.flush()
    bnt, labels = patsy_regression_weights(data, formula)
    print 'Doing PCA cleaning'
    sys.stdout.flush()
    X, Xpca, D = pca_cleaning(data, valid_conditions)
    print 'Regression embedding'
    sys.stdout.flush()
    Q, Bmax, t_bmax, norms, maps = regression_embedding(bnt, D)
    return Q, Bmax, labels, bnt, D, t_bmax, norms, maps


def valid_conditions(data, factors):
    conditions = []
    for c in dict_product(factors):
        p = conmean(data, **c)
        if sum(isnan(p)) > 0:
            continue
        conditions.append(c)
    return conditions


def get_trajectory(data, conditions, axis1, axis2):
    results = {}    
    for i, condition in enumerate(conditions):
        population_activity = condition_matrix(data, [condition])
        assert population_activity.shape[1] == data.data.shape[1]
        vax1 = dot(axis1, population_activity)
        vax2 = dot(axis2, population_activity)
        results[str(condition)] = (vax1, vax2)
    return results

def plot_population_activity(data, factors, axis1, axis2, 
        legend=False, epochs=None):
    import seaborn as sns
    conditions = list(dict_product(factors))
    colors = sns.color_palette('muted', len(conditions))
    symbols = ['-', '--', '-.', ':']
    leg = []
    for i, condition in enumerate(conditions):
        population_activity = condition_matrix(data, [condition])
        assert population_activity.shape[1] == data.data.shape[1]
        vax1 = dot(axis1, population_activity)
        vax2 = dot(axis2, population_activity)
        if epochs is None:
            h = plt.plot(vax1, vax2, color=colors[i])[0]
            plt.plot(vax1[0], vax2[0], 'o', color=colors[i])
            plt.plot(vax1[-1], vax2[-1], 'D', color=colors[i])
        else:
            for j, (low, high) in enumerate(epochs):
                h = plt.plot(vax1[low:high], 
                        vax2[low:high], ls=symbols[j%len(symbols)], 
                        color=colors[i])[0]
                plt.plot(vax1[low], vax2[low], 'o', color=colors[i])
                plt.plot(vax1[high], vax2[high], 'D', color=colors[i])

        leg.append((h, str(condition)))
    if legend:
        plt.legend([l[0] for l in leg], [l[1] for l in leg])
    return [l[0] for l in leg], [l[1] for l in leg] 


def trellis_plot(data, Q, labels, condition, Bmax=None):
    import matplotlib
    gs = matplotlib.gridspec.GridSpec(3, 3)
    for k in range(1,Q.shape[1]):
        for j in range(Q.shape[1]-1):
            plt.subplot(gs[k-1, j])
            if k <= j:
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('')
                plt.ylabel('')
                continue
            leg = plot_population_activity(
                data,
                condition,
                Q[:, k], Q[:, j])

            plt.xlabel('%s' % labels[k])
            plt.ylabel('%s' % labels[j])
            xmax = max(absolute(plt.xlim()))
            ymax = max(absolute(plt.ylim()))
            if Bmax is not None:
                a = dot(Q[:, [k,j]].T, Bmax[:, [k,j]])
                plt.plot([-a[1,0], a[1,0]], [-a[1,1], a[1,1]], 'k--')
                plt.plot([-a[0,0], a[0,0]], [-a[0,1], a[0,1]], 'k--')

                plt.axis('equal')
    plt.subplot(gs[0, 2])
    plt.legend(leg[0], leg[1])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

def dict_filter(dm, condition):
    idx = ones(dm.field(dm.fieldnames()[0]).shape).astype(bool)
    for key, val in condition.iteritems():
        idx = idx & (dm.field(key) == val)
    return dm[idx]


def characterize_population(data, bnt, factors, Q):
    conditions = list(dict_product(factors))
    gs = matplotlib.gridspec.GridSpec(
            len(conditions) + 1, len(unique(data.unit)))
    for i, cond in enumerate(conditions):
        d = dict_filter(data, cond)
        for iu, u in enumerate(d.by_field('unit')):
            plt.subplot(gs[i, iu])
            plt.plot(u.data.T.mean(0), 'k', alpha=0.5)
            plt.ylim([-max(absolute(plt.ylim())), max(absolute(plt.ylim()))])
            plt.plot(plt.xlim(), [0, 0], 'k--')
            plt.xticks([])
            plt.yticks([])
            if iu == 0:
                plt.ylabel(str(cond))
    for iu,unit in enumerate(unique(data.unit)):
        plt.subplot(gs[len(conditions), iu])
        bs = [bnt[unit, t] for t in range(bnt.shape[1])]
        plt.plot(array(bs)[:, (0,2)])
        plt.legend(factors.keys())




