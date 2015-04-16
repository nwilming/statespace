import h5py
from numpy import *
from ocupy import datamat
from itertools import product as iproduct
import glob, os
from pylab import *

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
    files = glob.glob(os.path.join(directory, substr+'*.mat'))
    print files
    outname = os.path.join(directory, substr + '_combined.dm')
    output_file = h5py.File(outname)
    try:
        offset = 0
        for i, f in enumerate(files):
            dm = datamat.load(f+'.datamat')
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
    combine_subjects('%s_cleandata'%subject, os.path.join(target_dir, 'minified'), 
        length, min(widths))


def analyze_subs(sub, epoche):
    import statespace as st
    factors = {'response':[-1, 1], 'stim_strength':[-1, 1]}
    valid_conditions = [{'response': -1, 'stim_strength': -1},
            {'response': 1, 'stim_strength': -1},
            {'response': -1, 'stim_strength': 1},
            {'response': 1, 'stim_strength': 1}]
    formula = 'response+stim_strength+1'
    epochs = {'stimulus1':[0, 460], 'stimulus2':[461, 461+460], 'response': [1+461+460, 461+460+148]}
    figure()
    key, value = epoche, epochs[epoche]
    print sub
    dm = datamat.load('/run/media/nwilming/dump/Data-Anne/minified/%s_combined.dm'%sub, 'datamat')
    dm.data = dm.data[:, value[0]:value[1]]
    st.zscore(dm)
    Q, Bmax, labels, bnt, D = st.embedd(dm, formula, valid_conditions)
    leg = st.plot_population_activity(dm, 
        {'response':[-1, 1], 'stim_strength':[-1,1]}, 
        Q[:, 1], Q[:, 2], epochs=[(0, dm.data.shape[1]-1)])
    title(key)
    del dm
    suptitle(sub)
    savefig('%s_%s.png'%(sub, epoche))

if __name__ == '__main__':
    import sys
    subject = sys.argv[1]
    preprocess_data(subject)
