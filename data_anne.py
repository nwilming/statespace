import h5py
from numpy import *
from ocupy import datamat
from itertools import product as iproduct
import glob, os, sys
from pylab import *
import seaborn as sns
import statespace as st 

conditions = ['stim_strength', 'response', 'choice', 'correct']

subjects = [12, 13, 15, 16, 17, 10, 9, 8, 2, 3, 6, 7, 4, 5, 20, 21, 14, 19, 18]
subject_files = {}
tfr_files = {}
for sub in subjects:    
    sessions =  glob.glob('/home/aurai/Data/MEG-PL/P%02i/MEG/Preproc/*cleandata.mat'%sub)
    subject_files[sub] = zip(range(len(sessions)), sessions)
    sessions = glob.glob('/home/aurai/Data/MEG-PL/P%02i/MEG/TFR/*fb_all_freq.mat'%sub)
    tfr_files[sub] = zip(range(len(sessions)), sessions)


def get_response_lock(num_samples):
    def response_lock(trial_info, trial_data, trial_time, units):
        response = trial_info[8].astype(int)
        data = trial_data[response-num_samples:response, :]
        time = trial_time[response-num_samples:response, 0]
        # The next line converts the response hand code from [12, 18] to [-1, 1]
        response_hand = (((trial_info[5] - 12)/6) - 0.5) *2        
        trial = {'stim_strength':trial_info[3], 'choice':trial_info[6], 'correct':trial_info[7],
                'response_hand':response_hand, 'time':time}
        for idx, unit in units:
            trial[unit] = data[:,idx]
        return trial

    return response_lock

def get_tfr_response_lock():
    def response_lock(trial_info, trial_data, trial_time, units):        
        data = trial_data
        time = trial_time
        # The next line converts the response hand code from [12, 18] to [-1, 1]
        response_hand = (((trial_info[5] - 12)/6) - 0.5) *2        
        trial = {'stim_strength':trial_info[3], 'choice':trial_info[6], 'correct':trial_info[7],
                'response_hand':response_hand, 'time':time}
        for idx, unit in units:
            trial[unit] = data[:,idx]
        return trial
    return response_lock

def adaptor(subject_files, select_data):
    '''
    Sits on top of a matlab file and returns a datamat to access it.

    select_data : function that receives trialinfo and trialdata field. It returns
        the data to be used for this trial and a dict containing metadata.
    '''
    trials = []
    for subject, data in subject_files.iteritems():
        for session, filename in data:
            data = h5py.File(filename) 
            labels = []
            for label in data['data']['label'][:].T:
                labels.append(''.join([unichr(t) for t in data[label[0]]]).encode('utf8'))

            trialinfo = data['data']['trialinfo'][:,:]        
            trial_data = data['data']['trial']
            trial_time = data['data']['time']
            channels = [(i, t) for i, t in zip(range(len(labels)), labels) if t.startswith('M')]
            for j, (td, ti, tt) in enumerate(zip(trial_data, trialinfo.T, trial_time)):            
                td_vals = data[td[0]]
                tt_vals = data[tt[0]]
                d = select_data(ti, td_vals, tt_vals, channels)
                d.update({'subject':array([subject])[0], 'session':array([session])[0]})
                trials.append(d)
                sys.stdout.flush()
    return trials, channels


def tfr_adaptor(subject_files, select_data, freq=None, struct='freq'):
    '''
    Sits on top of a matlab file and returns a datamat to access it.

    select_data : function that receives trialinfo and trialdata field. It returns
        the data to be used for this trial and a dict containing metadata.
    '''
    trials = []
    trial_id = 0
    for subject, data in subject_files.iteritems():
        for session, filename in data:
            print filename
            data = h5py.File(filename)
            labels = []
            for label in data[struct]['label'][:].T:
                labels.append(''.join([unichr(t) for t in data[label[0]]]).encode('utf8'))
            trialinfo = data[struct]['trialinfo'][:,:]        
            trial_data = data[struct]['powspctrm']
            trial_time = data[struct]['time']
            frequencies = data[struct]['freq']
            
            channels = [(i, t) for i, t in zip(range(len(labels)), labels) if t.startswith('M')]
            # trial_data is a four dimensional matrix:            
            # time x frequency x sensors x trial
            for trial_num in range(trial_data.shape[3]):
                if freq is None:
                    freq = zip(frequencies, frequencies)
                for i, (low, high) in enumerate(freq):
                    idx = (low<=frequencies) & (frequencies<=high)
                    d = select_data(trialinfo[:, trial_num],
                                    trial_data[:, idx, :, trial_num], trial_time[:].flatten(), channels)
                    d.update({'trial_id':array([trial_id])[0], 'subject':array([subject])[0], 'session':array([session])[0], 'freq':array([mean([low, high])])[0]})
                    trials.append(d)
                    trial_id += 1 
                sys.stdout.flush()
                    
    return trials, channels



def tolongform(trials, channels):
    ### Now convert to long form.
    length = len(channels)*len(trials)
    width = trials[0][channels[0][1]].shape[0]
    offset = 0
    fields = set(trials[0].keys()) - set([c[1] for c in channels]) - set(['time'])
    dm = {}
    for field in fields:
        print field, trials[0][field]
        dm[field] = empty((length,), dtype=trials[0][field].dtype)
    dm['data'] = nan*empty((length, width))
    dm['unit'] = empty((length,), dtype='S16')
    dm['time'] = 0*empty((length, width))
    trialnum = 0
    for trial in trials:
        trialnum+=2
        for idx, channel in channels:
            for field in fields:
                dm[field][offset] = trial[field]
            dm['data'][offset,:] = trial[channel]
            dm['time'][offset,:] = trial['time']
            dm['unit'][offset] = channel
            offset += 1 
    return datamat.VectorFactory(dm, {})

if __name__ == '__main__':
    task = sys.argv[1]
    sub = int(sys.argv[2])
    if task == 'pre':
        files = {sub:subject_files[sub]}
        trials, channels = adaptor(files, get_response_lock(650))
        dm = tolongform(trials, channels)
        dm.save('P%02i.datamat'%sub)
    elif task == 'tfr':
        files = {sub:tfr_files[sub]}
        trials, channels = tfr_adaptor(files, get_tfr_response_lock(), freq=[(0, 12), (13, 40)])
        dm = tolongform(trials, channels)
        dm.save('P%02i_tfr.datamat'%sub)

