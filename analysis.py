import h5py
from numpy import *
from ocupy import datamat
from itertools import product as iproduct
import glob, os
from pylab import *
import seaborn as sns
import statespace as st 

def analyze_subs(sub, channels=None, prefix=''):
    import statespace as st
    factors = {'response_hand':[-1, 1], 'stim_strength':[-1, 1]}
    valid_conditions = [{'response_hand': -1, 'stim_strength': -1},
            {'response_hand': 1, 'stim_strength': -1},
            {'response_hand': -1, 'stim_strength': 1},
            {'response_hand': 1, 'stim_strength': 1}]
    formula = 'response_hand+stim_strength+C(block)+1'
    dm = datamat.load('P%02i.datamat'%sub, 'Datamat')
    print dm
    if channels is not None:
        dm.data = dm.data[:,channels]
    # Need to identify no nan starting point
    a = array([st.conmean(dm, **v) for v in valid_conditions])
    try:
        idend = where(sum(isnan(a),0) > 0)[0][0]
        dm.data = dm.data[:, 0:idend]
    except IndexError:
        pass
    st.zscore(dm)
    Q, Bmax, labels, bnt, D, t_bmax, norms, maps = st.embedd(dm, formula, valid_conditions)
    results = st.get_trajectory(dm, 
        {'response_hand':[-1, 1], 'stim_strength':[-1,1]}, 
        Q[:, 1], Q[:, 2])
    del dm
    import cPickle
    results.update({'Q':Q, 'Bmax':Bmax, 'labels':labels,
        't_bamx':t_bmax, 'norms':norms, 'maps':maps,
        'factors':factors, 'valid_conditions':valid_conditions})
    cPickle.dump(results, open('%s%s.trajectory'%(prefix, sub), 'w'))


def combine_trajectories(trjs, select_samples=None):
    '''
    trjs is a (subject code, trajectory dict) tuple
    '''
    if select_samples is None:
        select_samples = lambda x: x
    dm = datamat.AccumulatorFactory()
    for subject, trj in trjs:
        trajectory = {'subject':subject}
        for j, (key, value) in enumerate(trj.iteritems()):
            trajectory['condition_%i'%j] = select_samples(value)
        dm.update(trajectory)
    return dm.get_dm(), trj.keys()

def tolongform(trjs, condition_mapping, select_samples=None):
    '''
    trjs is a (subject code, trajectory dict) tuple
    '''
    if select_samples is None:
        select_samples = lambda x: x
    conditions = condition_mapping.keys()
    print conditions
    #conditions = trjs[0][1].keys()
    dm = datamat.DatamatAccumulator()
    for cond_nr,  cond in enumerate(conditions):
        for subject, filename, trj in trjs:
            ax1 = select_samples(trj[cond][0])
            ax2 = select_samples(trj[cond][1]) 
            ax1label, ax2label = trj['labels'][1:]
            data = concatenate((ax1, ax2))
            axes = concatenate(([ax1label]*len(ax1), [ax2label]*len(ax1)))
            trial = {'subject':0*data+subject, 
                'condition':array([condition_mapping[cond]]*len(axes), dtype='S64'),                
                'data':data,
                'encoding_axis':axes,
                'time':concatenate((linspace(-len(ax1)/600., 0, len(ax1)), 
                    linspace(-len(ax1)/600., 0, len(ax1))))}
            dm.update(datamat.VectorFactory(trial,{}))
    dm = dm.get_dm()
    dm.add_field('used_hand', mod(dm.subject,2)==0)
    return dm, conditions



axes_labels = ['response', 'stimulus strength']

def make_1Dplot(df, encoding_axes=0):    
    sns.tsplot(df[df.encoding_axis==encoding_axes], time='time', unit='subject', 
            value='data', condition='condition')
    ylabel(encoding_axes)
    axhline(color='k')


def make_2Dplot(df):
    colors = sns.color_palette()
    conditions = []
    leg = []
    ax1label, ax2label = unique(df.encoding_axis)
    for i, (cond, df_c) in enumerate(df.groupby('condition', sort=False)):        
        ax1 = df_c[df_c.encoding_axis==ax1label].pivot('subject', 'time', 'data').values
        ax2 = df_c[df_c.encoding_axis==ax2label].pivot('subject', 'time', 'data').values        
        plot(ax1.mean(0), ax2.mean(0), color=colors[i])
        plot(ax1.mean(0)[0], ax2.mean(0)[0], color=colors[i], marker='s')
        plot(ax1.mean(0)[-1], ax2.mean(0)[-1], color=colors[i], marker='>')
        conditions.append(cond)
    axvline(color='k')
    axhline(color='k')
    legend(leg, conditions)
    xlabel(ax1label)
    ylabel(ax2label)
 

def get_conditions(conditions, files, w, condition_mapping):
    dm = datamat.DatamatAccumulator()
    for subject, file in files:        
        data = datamat.load(file, 'Datamat')
        st.zscore(data)
        for cond_nr, cond in enumerate(conditions):                    
            conmean = nanmean(st.condition_matrix(data, [cond]), 0)[-w:]            
            trial = {'subject':0*ones(conmean.shape)+subject, 
                'condition':array([condition_mapping[str(cond)]]*len(conmean)),
                'data':conmean,
                'time':linspace(-len(conmean)/600., 0, len(conmean))}
            dm.update(datamat.VectorFactory(trial,{}))
    dm = dm.get_dm()
    dm.add_field('used_hand', mod(dm.subject,2)==0)   
    return dm

if __name__ == '__main__':
    import sys
    task, subject = sys.argv[1:3]
    subject = int(subject)
    if task == 'analyze':
        analyze_subs(subject)
    elif task == 'analyze_occ':
        from scipy.io import loadmat
        channel_selection = loadmat('sensorselection.mat')['chans'][0,0][1].flatten()-1
        analyze_subs(subject, channel_selection, 'occ')
    elif task == 'analyze_motor':
        from scipy.io import loadmat
        channel_selection = loadmat('sensorselection.mat')['chans'][0,1][1].flatten()-1
        analyze_subs(subject, channel_selection, 'motor')

 
