
'''

Defines a function to perform constrained L1-norm minimization.  Unused.

TODO: delete

'''

from __future__ import division

import numpy as np

from scipy.optimize import linprog

def linear_least_l1_regression(A, b, G = None, h = None):
	# adapted from https://en.wikipedia.org/wiki/Least_absolute_deviations#Solving_using_linear_programming

	(n, m) = A.shape

	c = np.concatenate([
		np.zeros(m),
		np.ones(n)
		])

	I = np.identity(n)

	if G is None:
		G = np.empty((0, m))
		h = np.empty((0,))

	A_ub = np.concatenate([
		np.concatenate([-A, -I], 1),
		np.concatenate([+A, -I], 1),
		np.concatenate([+G, np.zeros((G.shape[0], n))], 1)
		],
		0)

	b_ub = np.concatenate([-b, +b, h])

	result = linprog(
		c,
		A_ub, b_ub,
		bounds = (None, None)
		)

	if not result.success:
		raise Exception(result.message)

	z = result.x
	f = result.fun

	x = z[:m]

	return x, f
