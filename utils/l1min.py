
from __future__ import division

import numpy as np

from scipy.optimize import linprog

def linear_least_l1_regression(A, y):
	# adapted from https://en.wikipedia.org/wiki/Least_absolute_deviations#Solving_using_linear_programming

	(n, m) = A.shape

	c = np.concatenate([
		np.zeros(m),
		np.ones(n)
		])

	I = np.identity(n)

	A_ub = np.concatenate([
		np.concatenate([-A, -I], 1),
		np.concatenate([+A, -I], 1)
		],
		0)

	b_ub = np.concatenate([-y, +y])

	result = linprog(
		c,
		A_ub, b_ub,
		bounds = (None, None)
		)

	if result.status != 0:
		raise Exception(result.message)

	z = result.x
	f = result.fun

	x = z[:m]

	return x, f
