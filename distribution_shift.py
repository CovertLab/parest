
from __future__ import division

import os

from itertools import izip

import numpy as np

import figure5
from ks_test import ks_test

(residuals_standard, indexing_standard) = figure5.get_residuals_and_indexing(
	os.path.join('out', 'all_scaled')
	)

(residuals_penalized, indexing_penalized) = figure5.get_residuals_and_indexing(
	os.path.join('out', 'all_scaled_upper_sat_limits_1e-1')
	)

assert np.all(indexing_standard == indexing_penalized)

indexing = indexing_standard.copy()

unique_values, unique_indices = np.unique(indexing, True)

# p_values = []

# for i in unique_indices:
# 	rs = residuals_standard[i]
# 	rp = residuals_penalized[i]
# 	p_values.append(ks_test(rs, rp)[1])

# sorting = np.argsort(p_values)

median_standard = np.median(residuals_standard[unique_indices, :], 1)
median_penalized = np.median(residuals_penalized[unique_indices, :], 1)

abs_diff = np.abs(median_penalized - median_standard)

sorting = np.argsort(abs_diff)

import utils.residuals

for s in sorting:

	if abs_diff[s] < utils.residuals.TENFOLD:
		continue

	(d, r, c) = unique_values[s]

	out = []

	if d != -1:
		out.append(figure5.DATATYPES_ORDERED[d])

	if r != -1:
		out.append(figure5.REACTIONS_ORDERED[r])

	if c != -1:
		out.append(figure5.COMPOUNDS_ORDERED[c])

	print ':'.join(out)
