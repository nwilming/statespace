import h5py
from numpy import *
from ocupy import datamat
from itertools import product as iproduct
import glob, os
from pylab import *
import seaborn as sns
import statespace as st 

conditions = ['stim_strength', 'response', 'choice', 'correct']

def simplify_data(filename, output, downsample=True):
    data = h5py.File(filename)
    trialinfo = data['data']['trialinfo'][:,:]
    epochs = {'stimulus_locked': (trialinfo[4,:].astype(int),
                                  trialinfo[8,:].astype(int))}
    trial_data = data['data']['trial']
    trial_time = data['data']['time']
    trials = []
    time = []
    conditions = []
    maxlen = max(epochs['stimulus_locked'][1] - epochs['stimulus_locked'][0]) 
    for j, (td, tt) in enumerate(zip(trial_data, trial_time)):
        data_d, data_t = [], []
        for epoch in ['stimulus_locked']:
            start, end = epochs[epoch]
            vals = data[td[0]][start[j]:end[j], :]
            vals = pad(vals, ((0, maxlen-vals.shape[0]), (0,0)), 
                    'constant', constant_values=(nan,nan))
            tvals = data[tt[0]][start[j]:end[j], :]
            tvals = pad(tvals, ((0, maxlen-tvals.shape[0]), (0,0)), 
                    'constant', constant_values=(nan,nan))
            data_d.append(vals)
            data_t.append(tvals)
        trials.append(concatenate(data_d))
        time.append(concatenate(data_t))
        # stim_strength', 'response', 'choice', 'correct'
        c = trialinfo[(3, 5, 6, 7), j]
        conditions.append(c)
    labels = []
    for label in data['data']['label'][:].T:
        labels.append(''.join([unichr(t) for t in data[label[0]]]).encode('utf8'))
    out_file = h5py.File(output, 'w')
    try:
        out_file.create_dataset('trials', data=array(trials))
        out_file.create_dataset('time', data=array(time))
        out_file.create_dataset('conditions', data=array(conditions))
        out_file.create_dataset('label', data=labels)
    finally:
        out_file.close()
        data.close()


def minify_subject(substr, directory, outputdir):
    files = glob.glob(os.path.join(directory, substr+'*.mat'))
    if len(files) == 0:
        raise RuntimeError('No Files found in %s with glob %s*.mat'%(directory, substr))
    print 'Minifying', files
    for filename in files:
        output = os.path.join(outputdir, 'minified', filename.replace(directory, ''))
        simplify_data(filename, output)


def subjects2datamat(substr, directory):
    files = glob.glob(os.path.join(directory, substr+'*.mat'))
    length = 0
    widths = []
    for i, filename in enumerate(files):
        print filename
        d = h5py.File(filename)
        try:
            trial_data = d['trials'][:, :, :]
            conditions = recode_regressors(d['conditions'][:,:])
            dm = data2dm(trial_data, conditions, d['time'][:], d['label'][:])
            dm.add_field('block', i*ones((len(dm),)))
            dm.save(filename + '.datamat')
            length += len(dm)
            widths.append(dm.data.shape[1])
        finally:
            d.close()
        del dm, trial_data, conditions
    return length, widths

def combine_subjects(substr, directory, length, width):
    files = glob.glob(os.path.join(directory, substr+'*.mat.datamat'))
    print files
    outname = os.path.join(directory, substr + '_combined.dm')
    output_file = h5py.File(outname)
    try:
        offset = 0
        for i, f in enumerate(files):
            dm = datamat.load(f)
            if i == 0:
                output_file.create_group('datamat')
                for f in dm.fieldnames():
                    shape = list(dm.field(f).shape)
                    if len(shape)==1:
                        shape[0] = length
                    else:
                        shape = length, width
                    output_file['datamat'].create_dataset(f, shape=shape)
            for f in dm.fieldnames():
                print f, dm.field(f).shape
                print output_file['datamat'][f].shape
                if len(dm.field(f).shape) == 1:
                    output_file['datamat'][f][offset:offset+len(dm)] = dm.field(f)
                else:
                    output_file['datamat'][f][offset:offset+len(dm), :] = dm.field(f)[:,:width]
            offset += len(dm)
            del dm
    finally:
        output_file.close()
    return outname


def data2dm(trial_data, trial_conditions, time, labels):
    '''
    This function combines data from a matlab file into a datamat.
    '''
    dm = datamat.AccumulatorFactory()
    idx = array([t.startswith('M') for t in labels])
    trial_data = trial_data[:, :, idx]
    for sensor, trial_id in iproduct(range(trial_data.shape[2]), range(trial_data.shape[0])):
        trial = {'data': trial_data[trial_id, :, sensor], 'time': time[trial_id, :, 0], 'unit': sensor}
        for key, value in zip(conditions, trial_conditions[trial_id, :]):
            trial[key] = value
        dm.update(trial)
    return dm.get_dm()


def recode_regressors(trial_conditions):
    trial_conditions[trial_conditions[:, 1] == 12, 1] = -1
    trial_conditions[trial_conditions[:, 1] == 18, 1] = 1
    trial_conditions[trial_conditions[:, 3] == 0, 3] = -1
    return trial_conditions

def preprocess_data(subject):
    source_dir = '/home/aurai/Data/MEG-PL/%s/MEG/Preproc/'%subject
    print source_dir
    target_dir = '/home/nwilming/data/anne_meg/'        
    minify_subject(subject+'*cleandata', source_dir, target_dir)
    length, widths = subjects2datamat('%s*cleandata'%subject, 
        os.path.join(target_dir, 'minified'))
    combine_subjects('%s'%subject, os.path.join(target_dir, 'minified'), 
        length, min(widths))



def analyze_subs(sub, channels=None, prefix=''):
    import statespace as st
    factors = {'response':[-1, 1], 'stim_strength':[-1, 1]}
    valid_conditions = [{'response': -1, 'stim_strength': -1},
            {'response': 1, 'stim_strength': -1},
            {'response': -1, 'stim_strength': 1},
            {'response': 1, 'stim_strength': 1}]
    formula = 'response+stim_strength+1'
    dm = datamat.load('/home/nwilming/data/anne_meg/minified/%s_combined.dm'%sub, 'datamat')
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
        {'response':[-1, 1], 'stim_strength':[-1,1]}, 
        Q[:, 1], Q[:, 2])
    del dm
    import cPickle
    results.update({'Q':Q, 'Bmax':Bmax, 'labels':labels,
        't_bamx':t_bmax, 'norms':norms, 'maps':maps,
        'factors':facotrs, 'valid_conditions':valid_conditions})
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
    for cond_nr, cond in enumerate(conditions):
        for subject, trj in trjs:
            ax1 = select_samples(trj[cond][0])
            ax2 = select_samples(trj[cond][1]) 
            data = concatenate((ax1, ax2))
            axes = concatenate((ax1*0, ax1*0+1))
            trial = {'subject':0*axes+subject, 
                'condition':array([condition_mapping[cond]]*len(axes), dtype='S64'),                
                'data':data,
                'encoding_axis':axes,
                'time':concatenate((linspace(-len(ax1)/600., 0, len(ax1)), 
                    linspace(-len(ax1)/600., 0, len(ax1))))}
            dm.update(datamat.VectorFactory(trial,{}))
    dm = dm.get_dm()
    dm.add_field('response_hand', mod(dm.subject,2)==0)
    return dm, conditions



axes_labels = ['choice', 'stimulus strength']

def make_1Dplot(df, encoding_axes=0):    
    sns.tsplot(df[df.encoding_axis==encoding_axes], time='time', unit='subject', 
            value='data', condition='condition')
    axhline(color='k')


def make_2Dplot(df):
    colors = sns.color_palette()
    conditions = []
    leg = []
    for i, (cond, df_c) in enumerate(df.groupby('condition', sort=False)):
        ax1 = df_c[df_c.encoding_axis==0].pivot('subject', 'time', 'data').values
        ax2 = df_c[df_c.encoding_axis==1].pivot('subject', 'time', 'data').values
        leg.append(plot(ax1.mean(0), ax2.mean(0), color=colors[i])[0])
        plot(ax1.mean(0)[0], ax2.mean(0)[0], color=colors[i], marker='o')
        plot(ax1.mean(0)[-1], ax2.mean(0)[-1], color=colors[i], marker='s')
        conditions.append(cond)
    axvline(color='k')
    axhline(color='k')
    legend(leg, conditions)
    xlabel(axes_labels[0])
    ylabel(axes_labels[1])
 

def get_conditions(conditions, files, w, condition_mapping):
    dm = datamat.DatamatAccumulator()
    for subject, file in files:        
        data = datamat.load(file, 'datamat')
        st.zscore(data)
        for cond_nr, cond in enumerate(conditions):                    
            conmean = nanmean(st.condition_matrix(data, [cond]), 0)[-w:]            
            trial = {'subject':0*ones(conmean.shape)+subject, 
                'condition':array([condition_mapping[str(cond)]]*len(conmean)),
                'data':conmean,
                'time':linspace(-len(conmean)/600., 0, len(conmean))}
            dm.update(datamat.VectorFactory(trial,{}))
    dm = dm.get_dm()
    dm.add_field('response_hand', mod(dm.subject,2)==0)   
    return dm

if __name__ == '__main__':
    import sys
    task, subject = sys.argv[1:3]
    if task == 'preprocess':
        preprocess_data(subject)
    elif task == 'analyze':
        analyze_subs(subject)
    elif task == 'analyze_occ':
        from scipy.io import loadmat
        channel_selection = loadmat('sensorselection.mat')['chans'][0,0][1].flatten()-1
        analyze_subs(subject, channel_selection, 'occ')
    elif task == 'analyze_motor':
        from scipy.io import loadmat
        channel_selection = loadmat('sensorselection.mat')['chans'][0,1][1].flatten()-1
        analyze_subs(subject, channel_selection, 'motor')

 
