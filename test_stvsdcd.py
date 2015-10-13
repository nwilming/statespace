from pylab import *

import sklearn, decoding, data_anne as da, pandas as pd, fake_experiment as fe
import statespace
import seaborn as sns
import numpy as np
from scipy.stats import circmean
sns.set_style('ticks')


# Test case one
def example(data):
    svm = lambda: sklearn.svm.SVC(kernel='linear')
    logistic = lambda: sklearn.linear_model.LogisticRegression(penalty='l2')
    np.random.seed(4)

    acc_color, color_mdls = decoding.do(decoding.chunker(data, 'session'), logistic, 'colorc', 'time')
    acc_mc, mc_mdls = decoding.do(decoding.chunker(data, 'session'), logistic, 'mc', 'time')
    d_color = decoding.accuracy(acc_color, level='time')
    d_mc = decoding.accuracy(acc_mc, level='time')

    subplot(2,2,1)
    sns.tsplot(d_color.T, color='c') # label='Color decoding'
    print d_color.T.shape
    sns.tsplot(d_mc.T, color='r') #label='MC decoding'
    ax1 = gca()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Decoding performance')
    ax1.set_ylim(-.1, 1.1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Decoding performance')
    ax1.set_ylim(-.1, 1.1)
    ax1.plot([0, 100], [0.5, 0.5], 'k')
    #for b in baselines:
    #    ax1.plot([0, 100], [b, b], 'k--')
    sns.despine(ax=ax1, top=True, left=False, right=True, bottom=False, offset=5)

    stmc, axesname, _ = statespace.cv_trajectory(decoding.chunker(data, 'session'),
                    'mc+colorc+1', ['mc'], time_label='time', filter=lambda x:x, N_components=2)
    stcolorc, axesname, Qs = statespace.cv_trajectory(decoding.chunker(data, 'session'),
                    'mc+colorc+1', ['colorc'], time_label='time', filter=lambda x:x, N_components=2)
    mc, colorc = stmc[1][:,0,:], stcolorc[1][:,1,:]

    subplot(2,2,2)
    plot(colorc.mean(0), color='c', label='color axis')
    plot(mc.mean(0), color='r',  label='mc axis')
    sns.tsplot(colorc, color='c') #label='color axis'
    sns.tsplot(mc, color='r') #label='mc axis'
    ax2=gca()
    ax2.set_ylabel('Axis weight')
    ax2.set_ylim(-.6, .6)
    ax2.plot([0, 100], [0, 0], 'r--', label='MC decoding')
    ax2.plot([0, 100], [0, 0], 'c--', label='Color decoding')
    ax2.plot([0, 100], [0., 0.], 'k')
    sns.despine(ax=ax2, top=True, left=False, right=True, bottom=False, offset=5)
    grid('off')
    legend()

    subplot(2,2,3)
    def angle(b,c):
        return arccos(dot(b,c)/(norm(b)*norm(c)))

    angles = array([[(angle(m.coef_, q.get_axis('mc')), angle(m.coef_, q.get_axis('colorc')))
            for (m, c, q) in zip(mcm, ccm, Qs)]
                    for time, (mcm, ccm) in enumerate(zip(array(mc_mdls).T, array(color_mdls).T))]).squeeze()
    print angles[:,:,:].shape
    def cm(x, **kwargs):
        kwargs['high']=pi
        return circmean(x, **kwargs)
    sns.tsplot(angles[:,:,0].T, color='r', estimator=cm)
    sns.tsplot(angles[:,:,1].T, color='c', estimator=cm)
    yticks([0, pi/4, pi/2, 3*pi/4, pi], rad2deg([0, pi/4, pi/2, 3*pi/4., pi]))
    return data, mc_mdls, color_mdls, Qs

def scattert(data, t, condition, ax_decode, ax_st):
    for cond, b in data.query('time==%i'%t).groupby(level=[condition]):
        color = 'b'
        if cond==1:
            color='r'
        scatter(b['data'][0], b['data'][1], color=color)
    plot([-ax_decode[0], ax_decode[0]], [-ax_decode[1], ax_decode[1]], 'k', label='Decoding')
    plot([-ax_st[0], ax_st[0]], [-ax_st[1], ax_st[1]], 'm', label='ST')
