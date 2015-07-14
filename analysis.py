import h5py
from numpy import *
from ocupy import datamat
from itertools import product as iproduct
import glob, os
from pylab import *
import seaborn as sns
import statespace as st 

def analyze_subs(input, output, channels=None, freq=None):
    import statespace as st
    factors = {'choice':[-1, 1], 'stim_strength':[-1, 1]}
    valid_conditions = [{'choice': -1, 'stim_strength': -1},
            {'choice': 1, 'stim_strength': -1},
           {'choice': -1, 'stim_strength': 1},
            {'choice': 1, 'stim_strength': 1}]
    #valid_conditions = [{'choice': -1},
    #        {'choice': 1}, {'stim_strength':1}, {'stim_strength':-1}]
 
    formula = 'choice+stim_strength+1'
    dm = datamat.load(input, 'Datamat')
    if freq is not None:
        if not freq in unique(dm.freq):
            raise RuntimeError('Selected frequency not in data. Available frequencies: ' + str(unique(dm.freq)))
        dm = dm[dm.freq==freq]
    if channels is not None:
        dm.data = dm.data[:,channels]
    # Need to identify no nan starting point
    a = array([st.conmean(dm, **v) for v in valid_conditions])
    try:
        idend = where(sum(isnan(a),0) > 0)[0][0]
        print 'The no nan end point is:', idend
        dm.data = dm.data[:, 0:idend]
    except IndexError:
        pass
    st.zscore(dm)
    Q, Bmax, labels, bnt, D, t_bmax, norms, maps = st.embedd(dm, formula, valid_conditions)
    results = st.get_trajectory(dm, 
        valid_conditions,
        Q[:, 1], Q[:, 2])
    del dm
    import cPickle
    results.update({'Q':Q, 'Bmax':Bmax, 'labels':labels,
        't_bamx':t_bmax, 'norms':norms, 'maps':maps,
        'factors':factors, 'valid_conditions':valid_conditions})
    cPickle.dump(results, open(output, 'w'))


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
    
    import sys, glob, os
    from optparse import OptionParser
    parser = OptionParser('python analze.py [-f] [-s] input output_trajectory')
    parser.add_option('--data-dir', dest='data_dir', default='data/')
    parser.add_option('--glob-str', dest='glob_string', default='P%02i_*freq.dm',
            help='Search pattern for globbing for subject data.')
    parser.add_option('-s', '--sensors', dest='sensor_selection', default=None, 
            help='Restrict to a set of sensors. Valid values are "occ" and "motor"')
    parser.add_option('-f', '--freq', dest='frequency', type=float, default=None, 
            help='Choose a frequency to analyze if you use tfr data.')
    parser.add_option('--dry-run', dest='dry_run', action='store_true', default=False)
    (options, args) = parser.parse_args()
    subject = int(args[0])
    channel_selection = None
    if options.sensor_selection == 'occ':
        from scipy.io import loadmat
        channel_selection = loadmat('sensorselection.mat')['chans'][0,0][1].flatten()-1
    elif options.sensor_selection == 'motor':
        from scipy.io import loadmat
        channel_selection = loadmat('sensorselection.mat')['chans'][0,1][1].flatten()-1
    else:
        if options.sensor_selection is not None:
            raise RuntimeError('did not understand the sensor selection argument. valid options are "occ" and "motor"')
    
    if '%' in options.glob_string:
        options.glob_string = options.glob_string%subject
    options.glob_string = os.path.join(options.data_dir, options.glob_string)
    input_files = glob.glob(options.glob_string)
    if options.frequency is None:
        output_files = [''.join(inp.split('.')[:-1]) + '.trajectory' for inp in input_files]
    else:
        output_files = [''.join(inp.split('.')[:-1]) + '_FR%3.1f_'%options.frequency + '.trajectory' for inp in input_files]

    print 'Using %s for globbing'%options.glob_string
    print 'Selected the following files for analysis:'
    for inp, out in zip(input_files, output_files):
        print inp, '->', out

    if not options.dry_run:
        for inp, out in zip(input_files, output_files):
            print 'Working'
            print inp, '->', out
            analyze_subs(inp, out, freq=options.frequency, channels=channel_selection)
 
