
from __future__ import division

import os

from itertools import izip

import numpy as np

import structure
import equations

SOURCE = os.path.join(
	'out',
	'all_scaled'
	# 'all_scaled_upper_sat_limits_1e-1'
	# 'all_scaled_upper_sat_limits_1e0'
	# 'all_scaled_upper_sat_limits_1e1'
	# 'all_scaled_upper_sat_limits_1e2'
	)

valid = np.load(os.path.join(SOURCE, 'valid.npy'))

pars = np.load(os.path.join(SOURCE, 'pars.npy'))[:, valid]

TARGET_FLUX = 0.14e-3

rrs = np.array([
	equations.reaction_rates(x, *equations.args)
	# np.log10(equations.reverse_reaction_rates(x, *equations.args)/equations.forward_reaction_rates(x, *equations.args))
	for x in pars.T
	])

def interquartile_range(values):
	return np.asscalar(np.diff(np.percentile(values, [25, 75])))

def median_absolute_deviation(values):
	return np.median(np.abs(values - np.median(values)))

SCALE = 1e-6

for name, rr in izip(structure.ACTIVE_REACTIONS, rrs.T):
	print '{}: {:0.2f} +- {:0.2f}, {:0.2%} +/- {:0.2%}'.format(
		name,
		np.median(rr / SCALE),
		median_absolute_deviation(rr / SCALE),
		np.median(rr / TARGET_FLUX),
		median_absolute_deviation(rr / TARGET_FLUX)
		)
