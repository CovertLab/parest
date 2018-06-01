
from __future__ import division

import os

from itertools import izip

import numpy as np
import matplotlib.pyplot as plt

import constants
import problems
import fitting

SOURCES = (
	os.path.join('out', 'all_scaled'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e-1'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e0'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e1'),
	os.path.join('out', 'all_scaled_upper_sat_limits_1e2'),
	)

NAMES = (
	'original',
	'penalty 1e-1',
	'penalty 1e0',
	'penalty 1e1',
	'penalty 1e2',
	)

COLORS = [
	np.array((225, 6, 133), np.float64)/255.,
	np.array((83, 49, 0), np.float64)/255.,
	np.array((143, 85, 0), np.float64)/255.,
	np.array((221, 145, 23), np.float64)/255.,
	np.array((251, 177, 37), np.float64)/255.,
	# np.array((0, 0, 255), np.float64)/255.,
	]

REFERENCE_RULES = problems.DEFINITIONS['all_scaled']

def get_absolute_fit_residuals(pars):
	(fm, fv, fe) = fitting.build_fitting_tensors(*REFERENCE_RULES)

	residuals = fm.dot(pars) - fv[:, None]

	return residuals

def get_relative_fit_residuals(pars):
	all_residuals = []

	for (fm, fv, fe) in fitting.build_relative_fitting_tensor_sets(*REFERENCE_RULES):
		raw_residuals = fm.dot(pars) - fv[:, None]

		medians = np.median(raw_residuals, 0)

		residuals = raw_residuals - medians[None, :]

		all_residuals.append(residuals)

	return np.row_stack(all_residuals)

def load_pars(directory):
	valid = np.load(os.path.join(directory, 'valid.npy'))
	pars = np.load(os.path.join(directory, 'pars.npy'))

	return pars[:, valid]

# def load_fit(directory):
# 	valid = np.load(os.path.join(directory, 'valid.npy'))
# 	obj = np.load(os.path.join(directory, 'obj.npy'))

# 	return obj[3, valid]

def load_lre(directory):
	valid = np.load(os.path.join(directory, 'valid.npy'))
	lre = np.load(os.path.join(directory, 'lre.npy'))

	return lre[valid]

def compute_fitness(pars):
	return (
		np.abs(get_absolute_fit_residuals(pars)).sum(0)
		+ np.abs(get_relative_fit_residuals(pars)).sum(0)
		)

plt.figure(figsize = (6, 6))

N_COLS = int(np.ceil(np.sqrt(len(SOURCES))))
N_ROWS = int(np.ceil(len(SOURCES) / N_COLS))

for (i, (name, source, color)) in enumerate(izip(NAMES, SOURCES, COLORS)):

	if i == 0:
		ax1 = plt.subplot(N_ROWS, N_COLS, i+1)

	else:
		plt.subplot(N_ROWS, N_COLS, i+1, sharex = ax1, sharey = ax1)

	crt = -constants.MU/load_lre(source)
	pars = load_pars(source)

	fit = compute_fitness(pars)

	plt.plot(
		crt, fit,
		'o',
		ms = 3, markeredgewidth = 0.5,
		markerfacecolor = color, markeredgecolor = 'w',
		label = name
		)

	plt.plot(
		np.median(crt), np.median(fit),
		'*',
		ms = 8, markeredgewidth = 1,
		markerfacecolor = 'w', markeredgecolor = 'k',
		zorder = 10
		)

	plt.title(name)

	plt.xscale('log')
	# plt.yscale('log')

	# plt.legend(loc = 'best')

	# plt.xlabel(r'Characteristic recovery time, relative to $\tau = 1 / \mu$')
	# plt.ylabel(r'Misfit error $f$')

	plt.xlabel('CRT')
	plt.ylabel('f')

plt.tight_layout()

plt.savefig('crt_vs_fit.pdf')
