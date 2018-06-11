
from __future__ import division

import os

import numpy as np

import constants
import equations

def median_absolute_deviation(values):
	return np.median(np.abs(values - np.median(values)))

sources = [
	os.path.join('out', 'history', 'naive'),
	os.path.join('out', 'history', 'standard'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e-1'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e0'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e1'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e2'),
	]

labels = []
values = [[] for i in xrange(len(sources))]

labels.append('n valid')
labels.append('percent valid')

for i, source in enumerate(sources):
	valid = np.load(os.path.join(source, 'valid.npy'))
	n_valid = valid.sum()
	fraction_valid = valid.mean()

	values[i].append('{:}'.format(n_valid))
	values[i].append('{:0.2%}'.format(fraction_valid))

labels.append('f median, MAD')

for i, source in enumerate(sources):
	fit = np.load(os.path.join(source, 'fit.npy'))[np.load(os.path.join(source, 'valid.npy'))]

	values[i].append('{:0.2f} +- {:0.2f}'.format(np.median(fit), median_absolute_deviation(fit)))

labels.append('CRT median, MAD')

for i, source in enumerate(sources):
	crt = -constants.MU / np.load(os.path.join(source, 'lre.npy'))[np.load(os.path.join(source, 'valid.npy'))]

	values[i].append('{:0.2e} +- {:0.2e}'.format(np.median(crt), median_absolute_deviation(crt)))

labels.append('percent pyk flux')

TARGET_FLUX = 0.14e-3

for i, source in enumerate(sources):
	pars = np.load(os.path.join(source, 'pars.npy'))[:, np.load(os.path.join(source, 'valid.npy'))]

	pyk_flux = np.array([
		equations.reaction_rates(p, *equations.args)[-2]/TARGET_FLUX
		for p in pars.T
		])

	values[i].append('{:0.2%} +- {:0.2%}'.format(np.median(pyk_flux), median_absolute_deviation(pyk_flux)))

for row in [labels] + values:
	print '\t'.join(row)
