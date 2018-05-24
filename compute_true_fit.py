
'''

Computes the *true* degree of fit to the training data, that is, without any
augmentation.

Usage:

python compute_true_fit.py <path to output directory>

Evaluate after calling gather_output.py.

Note: since the reference rules include no one-sided penalty terms, they have
been omitted here.  This will be an issue if the REFERENCE_RULES are changed to
include one-sided penalties.

'''

from __future__ import division

import os

from itertools import izip

import numpy as np
import matplotlib.pyplot as plt

import constants
import problems
import fitting

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
	pars = np.load(os.path.join(directory, 'pars.npy'))

	return pars

def compute_fitness(pars):
	return (
		np.abs(get_absolute_fit_residuals(pars)).sum(0)
		+ np.abs(get_relative_fit_residuals(pars)).sum(0)
		)

if __name__ == '__main__':
	import sys

	source = sys.argv[1]

	np.save(
		os.path.join(source, 'fit.npy'),
		compute_fitness(load_pars(source))
		)
