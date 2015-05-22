# Simulate some data.
# In each trial color has a certain effect, motion, choice and context.
from numpy import *
from random import choice as randsample
from ocupy import datamat, spline_base
from scipy.stats import norm
'''
Simulate an experiment similar to Mante 2013. A random dot display is
presented where the dots have a motion coherence and a color coherence.
Depending on the cue the monkeys have to attend either color or motion 
and they choose one of two options.
'''

time = arange(100)
motion_coh = array([-1, 1])
color_coh = array([-1, 1])
bfcts = [s for s in spline_base.spline_base1d(100, 10)[0].T]


def make_trial(motion, color, choice, context, trial, unit=None):
    '''
    Each unit has a template response to motion, color, choice, context.
    The template is a linear combination of gaussian basis functions.
    '''
    conditions = [randsample(motion_coh), randsample(
        color_coh), randsample([1, -1]), randsample([1, -1])]
    resp = 0*ones((100, ))
    for c_val, weights in zip(conditions, [motion, color, choice, context]):
        resp += c_val*array([r*b for r, b in zip(weights, bfcts)]).sum(0)
    resp = resp*linspace(0,1,len(resp))    
    return {'data': resp + random.randn(len(time)) * (resp.mean() / 5.), 'mc': conditions[0], 'colorc': conditions[1], 'choice': conditions[2],
            'ctxt': conditions[3], 'unit': unit, 'trial':trial}


def make_experiment(Ntrials, Nunits):
    units = datamat.AccumulatorFactory()
    for n in range(Nunits):
        # Each unit has a certain effect profile.
        weights = [random.randn(len(bfcts)) for _ in range(4)]
        _ = [
            units.update(
                make_trial(
                    *(weights+[trial]),
                    unit=n)) for trial in range(Ntrials)]

    return units.get_dm()

def make_test_case(Ntrials, Nreps):
    '''
    Construct a test case with two neurons where one neuron encodes
    motion coherence and the other neuron choice.

    '''
    units = datamat.AccumulatorFactory()
    for n in range(Nreps):
        unit_a = [arange(len(bfcts))] + [random.randn(len(bfcts))*4 for _ in range(3)]
        unit_b = [random.randn(len(bfcts))*4 for _ in range(4)]
        unit_b[2] = [0, 0, 1, 1, 1, 1, 1, 6, 6, 6, 6, 12, 12, 12]
        for _ in range(Ntrials):
            units.update(make_trial(*unit_a, unit=0+(n*2)))
            units.update(make_trial(*unit_b, unit=1+(n*2)))

    return units.get_dm()

