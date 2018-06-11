
'''

Performs the two-sample KS test, a non-parameteric test on whether two
distributions are significantly different.  Not used.

TODO: delete

'''


from __future__ import division

import numpy as np

def _edf(x, y):
	'''
	Evaluates the empirical density function on data points x at sample points
	y.
	'''
	xs = np.sort(x)
	f = (1 + np.arange(x.size)) / x.size

	out = np.empty(y.size, np.float64)

	too_small = y < xs[0]
	too_large = y >= xs[-1]

	out[too_small] = 0.0
	out[too_large] = 1.0

	in_range = ~too_small & ~too_large

	out_in_range = np.empty_like(out[in_range])

	(xi, yi) = np.where(np.diff(np.less_equal.outer(xs, y[in_range]), axis = 0))

	out_in_range[yi] = f[xi]

	out[in_range] = out_in_range

	return out

def ks_test(x1, x2):
	'''
	Returns the Kolmogorov-Smirnov test statistic and p-value on the empirical
	distribution functions given by data point x1 and x2.
	'''
	assert x1.ndim == 1
	assert x2.ndim == 1

	n1 = x1.size
	n2 = x2.size

	x_all = np.sort(np.concatenate([x1, x2]))

	f1 =_edf(x1, x_all)
	f2 =_edf(x2, x_all)

	largest = np.max(np.abs(f1 - f2))

	p_value = 2*np.exp(-np.square(largest / np.sqrt((n1 + n2)/(n1*n2)))*2)

	return largest, p_value
