
from __future__ import division

import os

from itertools import izip

import numpy as np
import matplotlib.pyplot as plt

import figure5
from ks_test import ks_test

import utils.residuals

OUTPUT_DIRECTORY = 'figure6_shifted_distributions'

if not os.path.exists(OUTPUT_DIRECTORY):
	os.mkdir(OUTPUT_DIRECTORY)

(residuals_standard, indexing_standard) = figure5.get_residuals_and_indexing(
	os.path.join('out', 'all_scaled')
	)

(residuals_small, indexing_small) = figure5.get_residuals_and_indexing(
	os.path.join('out', 'all_scaled_upper_sat_limits_1e-1')
	)

(residuals_large, indexing_large) = figure5.get_residuals_and_indexing(
	os.path.join('out', 'all_scaled_upper_sat_limits_1e2')
	)

assert np.all(indexing_standard == indexing_small)
assert np.all(indexing_standard == indexing_large)

indexing = indexing_standard

del indexing_standard, indexing_small, indexing_large

unique_values, unique_indices = np.unique(indexing, True)

median_standard = np.median(residuals_standard[unique_indices, :], 1)
median_small = np.median(residuals_small[unique_indices, :], 1)
median_large = np.median(residuals_large[unique_indices, :], 1)

abs_diff_small = np.abs(median_small - median_standard)
abs_diff_large = np.abs(median_large - median_standard)

shifted_small = abs_diff_small > utils.residuals.TENFOLD
shifted_large = abs_diff_large > utils.residuals.TENFOLD

assert np.all(shifted_large[shifted_small]), "Signficant shifts in the 'small' penalty optimizations not seen in the 'large' penalty optimizations."

plotted = np.in1d(indexing, unique_values[shifted_large])

for (name, residuals) in (
		('standard', residuals_standard),
		('small', residuals_small),
		('large', residuals_large)
		):

	fig = utils.residuals.plot(residuals[plotted], indexing[plotted])

	fig.savefig(os.path.join(OUTPUT_DIRECTORY, '{}.pdf'.format(name)), dpi = figure5.DPI)

	plt.close(fig)

with open(os.path.join(OUTPUT_DIRECTORY, 'key.txt'), 'w') as f:
	for uni in unique_values[shifted_large]:
		f.write(':'.join([
			figure5.DATATYPES_ORDERED[uni['datatype']] if uni['datatype'] >= 0 else '',
			figure5.REACTIONS_ORDERED[uni['reaction']] if uni['reaction'] >= 0 else '',
			figure5.COMPOUNDS_ORDERED[uni['compound']] if uni['compound'] >= 0 else '',
			])+'\n')
