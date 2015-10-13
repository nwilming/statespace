# Simulate some data.
# In each trial color has a certain effect, motion, choice and context.
from numpy import *
from random import choice as randsample
from ocupy import spline_base
import pandas as pd
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
bfcts = [s for s in spline_base.spline_base1d(max(time)+1, 10)[0].T]


def make_trial(motion, color, trial, conditions, unit=None, session=0):
    '''
    Each unit has a template response to motion, color, choice, context.
    The template is a linear combination of gaussian basis functions.
    '''

    resp = 0*time
    for c_val, weights in zip(conditions, [motion, color]):
        resp += c_val*array([r*b for r, b in zip(weights, bfcts)]).sum(0)
    x = linspace(0., 1, len(resp))
    y = 1-(1-x)**2 +  1-(x-0.5)**2
    y = y/max(y)

    resp = resp* y + random.randn(len(time)) * max(abs(resp.mean()), 0.125)
    return [{'data':r, 'mc': conditions[0], 'colorc': conditions[1],
             'unit': unit, 'trial':trial, 'time':t, 'session':session} for t, r in enumerate(resp)]


def make_experiment(Ntrials, Nunits, Nsessions):
    from itertools import product
    units = []
    unit_weights = {}
    conditions = {}
    trial_list = list(product(motion_coh, color_coh))*Ntrials
    for trial, vals in enumerate(trial_list):
        conditions[trial] = vals
    for n in range(Nunits):
        # Each unit has a certain effect profile.
        weights = [random.randn(len(bfcts)) for _ in range(2)]
        weights[0][-1] *= 0.8
        weights[1][-1] *= 0.8
        unit_weights[n] = weights

    for session, unit, trial in product(range(Nsessions), range(Nunits), range(Ntrials)):
        condition = conditions[trial]
        units.extend(make_trial(
                                *(unit_weights[unit]+[trial] + [condition]),
                                unit=unit, session=session))
    data = pd.DataFrame(units)
    data = data.set_index(['mc', 'colorc', 'unit', 'trial', 'session', 'time'])

    return data.unstack('unit'), array([v for v in conditions.values()])

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
