import h5py
from numpy import *
from ocupy import datamat
import glob
import os
import statespace as st
import cPickle


valid_conditions = [{'choice': -1, 'stim_strength': -1},
                    {'choice': 1, 'stim_strength': -1},
                    {'choice': -1, 'stim_strength': 1},
                    {'choice': 1, 'stim_strength': 1}]

# valid_conditions = [{'choice': -1},
#   {'choice': 1}, {'stim_strength':1}, {'stim_strength':-1}]


def _load_and_prep_data(input, channels, freq=None):
    '''
    Loads a datamat, filters by channels, averages across frequencies if needed
    '''
    dm = datamat.load(input, 'Datamat')
    if freq is not None:
        if freq not in unique(dm.freq):
            raise RuntimeError('Selected frequency not in data. Available frequencies: ' + str(unique(dm.freq)))
        dm = dm[dm.freq == freq]
    if channels is not None:

        idx = in1d(dm.unit, channels)
        dm = dm[idx]
        assert len(unique(dm.unit)) == len(channels)
    # Need to identify no nan starting point
    '''
    a = array([st.conmean(dm, **v) for v in valid_conditions])
    try:
        idend = where(sum(isnan(a),0) > 0)[0][0]
        print 'The no nan end point is:', idend
        dm.data = dm.data[:, 0:idend]
    except IndexError:
        pass
    '''
    return dm


def find_embedding(dm, valid_conditions, formula='choice+stim_strength+1'):
    '''
    Find a subspace embedding for data set dm.
    '''
    st.zscore(dm)
    Q, Bmax, labels, bnt, D, t_bmax, norms, maps, exp_var = st.embedd(dm,
        formula, valid_conditions, N_components=3)
    return {'Q': Q, 'Bmax': Bmax, 'labels': labels,
            't_bmax': t_bmax, 'norms': norms, 'maps': maps, 'D': D,
            'valid_conditions': valid_conditions, 'exp_var': exp_var}


def apply_embedding(dm, Q=None, Bmax=None, labels=None, D=None, t_bmax=None,
                    norms=None, maps=None, exp_var=None, **kwargs):
    '''
    Apply an embedding to a dataset to generate a state space trajectory.
    '''
    st.zscore(dm)
    # Make sure we are dealing with trials that are timelocked.
    assert sum(dm.time-dm.time[0]) <= finfo(float).eps
    results = st.get_trajectory(dm,
                                valid_conditions,
                                Q[:, 1], Q[:, 2], time=dm.time[0])
    results.update({'Q': Q, 'Bmax': Bmax, 'labels': labels,
                    't_bmax': t_bmax, 'norms': norms, 'maps': maps, 'D': D,
                    'valid_conditions': valid_conditions, 'exp_var': exp_var})
    return results


def combine_trajectories(trjs, select_samples=None):
    '''
    trjs is a (subject code, trajectory dict) tuple
    '''
    if select_samples is None:
        select_samples = lambda x: x
    dm = datamat.AccumulatorFactory()
    for subject, trj in trjs:
        trajectory = {'subject': subject}
        for j, (key, value) in enumerate(trj.iteritems()):
            trajectory['condition_%i' % j] = select_samples(value)
        dm.update(trajectory)
    return dm.get_dm(), trj.keys()


def tolongform(trjs, condition_mapping, axislabels, select_samples=None):
    '''
    trjs is a (subject code, trajectory dict) tuple
    '''
    if select_samples is None:
        select_samples = lambda x: x
    conditions = condition_mapping.keys()
    dm = datamat.DatamatAccumulator()
    for cond_nr,  cond in enumerate(conditions):
        for subject, filename, trj in trjs:
            ax1 = select_samples(trj[cond][0])
            ax2 = select_samples(trj[cond][1])
            time = select_samples(trj[cond][2])
            ax1label, ax2label = axislabels
            data = concatenate((ax1, ax2))
            axes = concatenate(([ax1label]*len(ax1), [ax2label]*len(ax1)))
            time = concatenate((time, time))
            trial = {'subject': 0*data+subject,
                     'condition': array(
                                        [condition_mapping[cond]]*len(axes),
                                        dtype='S64'),
                     'data': data,
                     'encoding_axis': axes,
                     'time': time}

            dm.update(datamat.VectorFactory(trial, {}))
    dm = dm.get_dm()
    dm.add_field('used_hand', mod(dm.subject, 2) == 0)
    return dm, conditions


def get_conditions(conditions, files, w, condition_mapping):
    '''
    I don't remember what this is doing. Maybe compute condition averages?
    '''
    dm = datamat.DatamatAccumulator()
    for subject, file in files:
        data = datamat.load(file, 'Datamat')
        st.zscore(data)
        for cond_nr, cond in enumerate(conditions):
            conmean = nanmean(st.condition_matrix(data, [cond]), 0)[-w:]
            trial = {'subject': 0*ones(conmean.shape)+subject,
                     'condition': array(
                                        [condition_mapping[str(cond)]] *
                                        len(conmean)),
                     'data': conmean,
                     'time': linspace(-len(conmean)/600., 0, len(conmean))}
            dm.update(datamat.VectorFactory(trial, {}))
    dm = dm.get_dm()
    dm.add_field('used_hand', mod(dm.subject, 2) == 0)
    return dm

if __name__ == '__main__':

    from optparse import OptionParser
    parser = OptionParser('python analze.py subject_number')
    parser.add_option('--data-dir', dest='data_dir', default='data/')
    parser.add_option('--glob-str', dest='glob_string',
                      default='P%02i_*freq.dm',
                      help='Search pattern for globbing for subject data.')
    parser.add_option('-s', '--sensors', dest='sensor_selection',
                      default=None,
                      help='Restrict to a set of sensors. Valid values are "occ" and "motor"')
    parser.add_option('-f', '--freq', dest='frequency',
                      type=float, default=None,
                      help='Choose a frequency to analyze if you use tfr data.')
    parser.add_option('-Q', '--apply-q', dest='applyQ',
                      type=str, default=None,
                      help='Specify a data file from which Q matrix is loaded. This needs to be a trajectory estimate from a previous run.')
    parser.add_option('--dry-run', dest='dry_run', action='store_true',
                      default=False)
    parser.add_option('--suffix', dest='suffix', default='',
                      help='Suffix to append to trajectory')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.error('Subject number is required')
    subject = int(args[0])
    channel_selection = None
    if options.sensor_selection == 'occ':
        channel_selection = [
                'MLP41', 'MPL31', 'MZP01', 'MRP31', 'MRP41', 'MLP53', 'MLP52',
                'MPL51', 'MRP51', 'MRP52', 'MRP53', 'MLO12', 'MLO11', 'MZO01',
                'MRO11', 'MRO12', 'MLO23', 'MLO22', 'MLO21', 'MRO21', 'MRO22',
                'MRO23', 'MLO32', 'MLO31', 'MZO02', 'MRO31', 'MRO32', 'MRP56',
                'MRP55', 'MRP54', 'MRP44', 'MRP43', 'MRP42', 'MRP34', 'MRP33',
                'MRP32', 'MRP22', 'MRP21', 'MRP11', 'MRP31', 'MZC04', 'MLP56',
                'MLP55', 'MLP54', 'MLP44', 'MLP43', 'MLP42', 'MLP34', 'MLP33',
                'MLP32', 'MLP22', 'MLP21', 'MLP11', 'MLP31']
    elif options.sensor_selection == 'motor':
        channel_selection = [
                'MLC21', 'MLC22', 'MLC52', 'MLC41', 'MLC23', 'MLC53', 'MLC31',
                'MLC24', 'MLC16', 'MLC25', 'MLC32', 'MLC42', 'MLC54', 'MLC55',
                'MLP12', 'MLP23', 'MRC21', 'MRC22', 'MRC52', 'MRC41', 'MRC23',
                'MRC53', 'MRC31', 'MRC24', 'MRC16', 'MRC25', 'MRC32', 'MRC42',
                'MRC54', 'MRC55', 'MRP12', 'MRP23']
    else:
        if options.sensor_selection is not None:
            raise RuntimeError('did not understand the sensor selection argument. valid options are "occ" and "motor"')

    if '%' in options.glob_string:
        options.glob_string = options.glob_string % subject
    options.glob_string = os.path.join(options.data_dir, options.glob_string)
    input_files = glob.glob(options.glob_string)
    if options.frequency is None:
        output_files = [''.join(inp.split('.')[:-1]) +
                        '%s.trajectory' % options.suffix for inp in input_files]
    else:
        output_files = [''.join(inp.split('.')[:-1]) +
                        '_FR%3.1f_' % options.frequency +
                        '%s.trajectory' % options.suffix for inp in input_files]

    print 'Using %s for globbing'%options.glob_string
    print 'Selected the following files for analysis:'
    for inp, out in zip(input_files, output_files):
        print inp, '->', out
    if options.applyQ is None:
        print 'Estimating Q embedding matrix for each of these files.'
    else:
        print 'Loading Q matrix from file. Applying this embedding to the above files.', options.applyQ
        embedding = cPickle.load(open(options.applyQ))

    if not options.dry_run:
        for inp, out in zip(input_files, output_files):
            print 'Working'
            print inp, '->', out
            dm = _load_and_prep_data(inp, channel_selection, freq=options.frequency)
            if options.applyQ is None:
                results = find_embedding(dm, valid_conditions)
            else:
                results = apply_embedding(dm, **embedding)
            cPickle.dump(results, open(out, 'w'))
