
from __future__ import division

import numpy as np
from scipy.optimize import minimize, linprog

import bounds
from utils.linalg import nullspace_projector
# from utils.l1min import linear_least_l1_regression

import structure

def build_initial_parameter_values(fitting_tensors, relative_fitting_tensor_sets = ()):

	(fitting_matrix, fitting_values) = fitting_tensors[:2]

	init_bounded = (bounds.LOWERBOUNDS + bounds.UPPERBOUNDS)/2

	G = np.concatenate([
		-bounds.BOUNDS_MATRIX,
		+bounds.BOUNDS_MATRIX
		])

	h = np.concatenate([
		-bounds.LOWERBOUNDS,
		bounds.UPPERBOUNDS
		]) - 1e-6 # tighten the bounds slightly
	# I believe this to be an issue with the precision of the solver

	if fitting_matrix.size == 0:
		res = minimize(
			lambda x: np.sum(np.square(
				bounds.BOUNDS_MATRIX.dot(x) - init_bounded
				)),
			np.zeros(structure.n_parameters),
			constraints = dict(
				type = 'ineq',
				fun = lambda x: h - G.dot(x)
				)
			)

		assert res.success

		init_pars = res.x

		fitting_residuals = np.empty((0,))

		fitness = 0.0

	else:

		SCALE = 1e-1

		(n_abs, n_pars) = fitting_matrix.shape
		n_rel_sets = len(relative_fitting_tensor_sets)

		def ones_col(index, n_rows, n_cols):
			out = np.zeros((n_rows, n_cols))

			out[:, index] = 1

			return out

		A = np.concatenate([
			np.concatenate([fitting_matrix, np.zeros([n_abs, n_rel_sets])], 1),
			] + [
			np.concatenate([fm, -ones_col(i, fv.size, n_rel_sets)], 1)
			for (i, (fm, fv, fe)) in enumerate(relative_fitting_tensor_sets)
			], 0)
		b = np.concatenate(
			[fitting_values]
			+ [fv for (fm, fv, fe) in relative_fitting_tensor_sets],
			0
			)

		# from utils.l1min import linear_least_l1_regression
		# fit_pars, fitness = linear_least_l1_regression(SCALE*A, SCALE*b, G, h)

		(n, m) = A.shape

		c = np.concatenate([
			np.zeros(m),
			np.ones(n)
			])

		I = np.identity(n)

		G = np.concatenate([
			G, np.zeros((G.shape[0], n_rel_sets))
			], 1)

		A_scaled = SCALE*A
		b_scaled = SCALE*b

		A_ub = np.concatenate([
			np.concatenate([-A_scaled, -I], 1),
			np.concatenate([+A_scaled, -I], 1),
			np.concatenate([+SCALE*G, np.zeros((G.shape[0], n))], 1)
			],
			0)

		b_ub = np.concatenate([-b_scaled, +b_scaled, SCALE*h])

		result = linprog(
			c,
			A_ub, b_ub,
			bounds = (None, None)
			)

		if not result.success:
			raise Exception(result.message)

		z = result.x
		fitness = result.fun

		fit_pars = z[:m]

		fitness /= SCALE

		fitting_residuals = A.dot(fit_pars) - b

		bounds_matrix = np.concatenate([
			bounds.BOUNDS_MATRIX,
			np.zeros((bounds.BOUNDS_MATRIX.shape[0], n_rel_sets))
			], 1)

		assert (
			(bounds_matrix.dot(fit_pars) >= bounds.LOWERBOUNDS)
			& (bounds_matrix.dot(fit_pars) <= bounds.UPPERBOUNDS)
			).all(), 'fit parameters not within bounds'

		N = nullspace_projector(A)

		res = minimize(
			lambda z: np.sum(np.square(
				bounds_matrix.dot(fit_pars + N.dot(z)) - init_bounded
				)),
			np.zeros(N.shape[1]),
			constraints = dict(
				type = 'ineq',
				fun = lambda z: h - G.dot(fit_pars + N.dot(z))
				)
			)

		assert res.success

		z = res.x

		init_pars = fit_pars + N.dot(z)

		init_fit = np.abs(
			fitness
			- np.sum(np.abs(A.dot(init_pars) - b))
			)

		assert init_fit < 1e-6, 'init parameters not fit'

		init_pars = init_pars[:n_pars]

	assert (
		(bounds.BOUNDS_MATRIX.dot(init_pars) >= bounds.LOWERBOUNDS)
		& (bounds.BOUNDS_MATRIX.dot(init_pars) <= bounds.UPPERBOUNDS)
		).all(), 'init parameters not within bounds'

	return (init_pars, fitness, fitting_residuals)

if __name__ == '__main__':
	import fitting
	import problems

	for problem, definition in problems.DEFINITIONS.viewitems():
		fitting_tensors = fitting.build_fitting_tensors(*definition)
		relative_fitting_tensor_sets = fitting.build_relative_fitting_tensor_sets(*definition)

		try:
			(init_pars, fitness, residuals) = build_initial_parameter_values(
				fitting_tensors, relative_fitting_tensor_sets
				)

		except Exception as e:
			print 'Failed to initialize problem "{}" with exception {}'.format(
				problem,
				e
				)
