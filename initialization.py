
from __future__ import division

import numpy as np
from scipy.optimize import minimize, linprog

# import bounds
from utils.linalg import nullspace_projector
# from utils.l1min import linear_least_l1_regression

import structure

BOUNDS_TOLERANCE = 1e-6 # used to slightly tighten the bounds to account for small numerical errors

LP_SCALE = 1e-2 # linear program problem scale

FIT_TOLERANCE = 1e-6 # acceptable adjustment to fit following second stage

def build_initial_parameter_values(
		fitting_tensors, relative_fitting_tensor_sets,
		bounds_matrix, lowerbounds, upperbounds
		):

	(fitting_matrix, fitting_values) = fitting_tensors[:2]

	init_bounded = (lowerbounds + upperbounds)/2

	G = np.concatenate([
		-bounds_matrix,
		+bounds_matrix,
		])

	h = np.concatenate([
		-lowerbounds,
		upperbounds,
		]) - BOUNDS_TOLERANCE

	if fitting_matrix.size == 0:
		raise Exception('this has not been tested in a long time, and does not properly check for relative fitting stuff')
		res = minimize(
			lambda x: np.sum(np.square(
				bounds_matrix.dot(x) - init_bounded
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

		(n_abs, n_pars) = fitting_matrix.shape
		n_rel_sets = len(relative_fitting_tensor_sets)

		def ones_col(index, n_rows, n_cols):
			out = np.zeros((n_rows, n_cols))

			out[:, index] = 1

			return out

		A = np.concatenate(
			[
				np.concatenate([
					fitting_matrix,
					np.zeros([n_abs, n_rel_sets])],
					1),
				]
			+ [
				np.concatenate([
					fm, -ones_col(i, fv.size, n_rel_sets)
					], 1)
					for (i, (fm, fv, fe)) in enumerate(relative_fitting_tensor_sets)
				],
			0
			)
		b = np.concatenate(
			[
				fitting_values
				]
			+ [
				fv for (fm, fv, fe) in relative_fitting_tensor_sets
				],
			0
			)

		(n, m) = A.shape

		c = np.concatenate([
			np.zeros(m),
			np.ones(n)
			])

		I = np.identity(n)

		G = np.concatenate([
			G, np.zeros((G.shape[0], n_rel_sets))
			], 1)

		A_scaled = LP_SCALE*A
		b_scaled = LP_SCALE*b

		A_ub = np.concatenate([
			np.concatenate([-A_scaled, -I], 1),
			np.concatenate([+A_scaled, -I], 1),
			np.concatenate([+LP_SCALE*G, np.zeros((G.shape[0], n))], 1),
			],
			0)

		b_ub = np.concatenate([
			-b_scaled,
			+b_scaled,
			LP_SCALE*h,
			])

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

		fitness /= LP_SCALE

		fitting_residuals = A.dot(fit_pars) - b

		augmented_bounds_matrix = np.concatenate([
			bounds_matrix,
			np.zeros((bounds_matrix.shape[0], n_rel_sets))
			], 1)

		assert (
			(augmented_bounds_matrix.dot(fit_pars) >= lowerbounds)
			& (augmented_bounds_matrix.dot(fit_pars) <= upperbounds)
			).all(), 'fit parameters not within bounds'

		N = nullspace_projector(A)

		res = minimize(
			lambda z: np.sum(np.square(
				augmented_bounds_matrix.dot(fit_pars + N.dot(z)) - init_bounded
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

		assert init_fit < FIT_TOLERANCE, 'init parameters not fit'

		init_pars = init_pars[:n_pars]

	assert (
		(bounds_matrix.dot(init_pars) >= lowerbounds)
		& (bounds_matrix.dot(init_pars) <= upperbounds)
		).all(), 'init parameters not within bounds'

	return (init_pars, fitness, fitting_residuals)

def test():
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

if __name__ == '__main__':
	test()
