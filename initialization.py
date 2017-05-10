
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

def column_of_ones_matrix(column, n_rows, n_columns, dtype = None):
	out = np.zeros((n_rows, n_columns), dtype)

	out[:, column] = 1

	return out

def compose_block_matrix(list_of_lists_of_matrices):
	return np.concatenate([
		np.concatenate(list_of_matrices, 1)
		for list_of_matrices in list_of_lists_of_matrices
		])

def build_initial_parameter_values2( # TODO: meaningful defaults
		init_matrix, init_values, # target values secondary to penalties
		upper_bounds_matrix, upper_bounds_values, # linear bounding
		absolute_penalty_matrix, absolute_penalty_values,
		upper_penalty_matrix, upper_penalty_values,
		*relative_penalty_matrices_and_values # matrix-value pairs
		):

	# TODO: documentation
	# TODO: thorough error checking on dimensions of inputs and outputs

	# Stage 1: set up and solve the linear program

	n_basic_bounds = upper_bounds_values.size
	n_absolute_penalties = absolute_penalty_values.size
	n_upper_penalties = upper_penalty_values.size
	n_each_relative_penalties = [
		relative_penalty_matrix.shape[0]
		for relative_penalty_matrix, relative_penalty_values
		in relative_penalty_matrices_and_values
		]
	n_relative_penalties = sum(n_each_relative_penalties)
	n_relative_penalty_sets = len(relative_penalty_matrices_and_values)

	n_basic_parameters = init_matrix.shape[1]
	n_free_parameters = n_relative_penalty_sets
	n_hidden_parameters = (
		n_absolute_penalties
		+ n_upper_penalties
		+ n_relative_penalties
		)

	n_parameters = (
		n_basic_parameters
		+ n_free_parameters
		+ n_hidden_parameters
		)

	n_hidden_bounds = 2*n_hidden_parameters

	n_bounds = (
		n_basic_bounds
		+ n_hidden_bounds
		)

	# following basic LP abstraction names:
	# c: objective coefficients
	# G, h: linear inequality constraints s.t. Gx <= h
	# there is no equality constraints i.e. Ax = b

	I_abs = np.identity(n_absolute_penalties)
	I_upper = np.identity(n_upper_penalties)
	I_rel = np.identity(n_relative_penalties)

	endat = np.cumsum(n_each_relative_penalties)
	startat = np.concatenate([[0], endat])[:-1]

	G_blocks = [
		# Basic upper bounds
		[
			upper_bounds_matrix,
			np.zeros((n_basic_bounds, n_free_parameters+n_hidden_parameters))
			],

		# Absolute penalties
		[
			-absolute_penalty_matrix,
			np.zeros((n_absolute_penalties, n_free_parameters)),
			-I_abs,
			np.zeros((n_absolute_penalties, n_upper_penalties+n_relative_penalties))
			],
		[
			+absolute_penalty_matrix,
			np.zeros((n_absolute_penalties, n_free_parameters)),
			-I_abs,
			np.zeros((n_absolute_penalties, n_upper_penalties+n_relative_penalties))
			],

		# Penalties for values above some threshold
		[
			+upper_penalty_matrix,
			np.zeros((n_upper_penalties, n_free_parameters+n_absolute_penalties)),
			-I_upper,
			np.zeros((n_upper_penalties, n_relative_penalties))
			],
		[
			np.zeros_like(upper_penalty_matrix),
			np.zeros((n_upper_penalties, n_free_parameters+n_absolute_penalties)),
			-I_upper,
			np.zeros((n_upper_penalties, n_relative_penalties))
			],
		] + [
		# Penalties for relative errors (lower)
		[
			-relative_penalty_matrix,
			+column_of_ones_matrix(i, relative_penalty_matrix.shape[0], n_relative_penalty_sets),
			np.zeros((relative_penalty_matrix.shape[0], n_absolute_penalties+n_upper_penalties)),
			-I_rel[startat[i]:endat[i], :]
			]
		for i, (relative_penalty_matrix, relative_penalty_values)
		in enumerate(relative_penalty_matrices_and_values)
		] + [
		# Penalties for relative errors (upper)
		[
			+relative_penalty_matrix,
			-column_of_ones_matrix(i, relative_penalty_matrix.shape[0], n_relative_penalty_sets),
			np.zeros((relative_penalty_matrix.shape[0], n_absolute_penalties+n_upper_penalties)),
			-I_rel[startat[i]:endat[i], :]
			]
		for i, (relative_penalty_matrix, relative_penalty_values)
		in enumerate(relative_penalty_matrices_and_values)
		]

	G = compose_block_matrix(G_blocks)

	h = np.concatenate([
		upper_bounds_values,
		-absolute_penalty_values,
		+absolute_penalty_values,
		+upper_penalty_values,
		np.zeros(n_upper_penalties),
		] + [
		-relative_penalty_values
		for relative_penalty_matrix, relative_penalty_values
		in relative_penalty_matrices_and_values
		] + [
		+relative_penalty_values
		for relative_penalty_matrix, relative_penalty_values
		in relative_penalty_matrices_and_values
		])

	c = np.concatenate([
		np.zeros(n_basic_parameters + n_free_parameters),
		np.ones(n_hidden_parameters)
		])

	result_linprog = linprog(
		c,
		G, h,
		bounds = (None, None),
		options = dict(
			tol = 1e-6
			)
		)

	z0 = result_linprog.x
	f = result_linprog.fun

	assert result_linprog.success

	# Step 2: regularize the choice of x in the nullspace of the solution for the LP
	# this isn't perfect, but it's not terribly important either
	# arguably all regularization should be implemented as some sort of (low value) penalty term

	# N = nullspace_projector(G) # shouldn't be G - need to fix!
	# N[np.abs(N) < 1e-3] = 0

	# A = np.concatenate(
	# 	[
	# 		init_matrix,
	# 		np.zeros((init_values.size, n_free_parameters+n_hidden_parameters))
	# 		],
	# 	1
	# 	)
	# b = init_values

	# h2 = upper_bounds_values
	# G2 = np.concatenate(
	# 	[
	# 		upper_bounds_matrix,
	# 		np.zeros((upper_bounds_values.size, n_free_parameters+n_hidden_parameters))
	# 		],
	# 	1
	# 	)

	# result_minimize = minimize(
	# 	lambda z: np.sum(np.square(
	# 		A.dot(z0 + N.dot(z)) - b
	# 		)),
	# 	np.zeros(n_parameters),
	# 	constraints = dict(
	# 		type = 'ineq',
	# 		fun = lambda z: h2 - G2.dot(z0 + N.dot(z))
	# 		)
	# 	)

	# z = z0 + N.dot(result_minimize.x)
	# x = z[:n_basic_parameters]

	# assert result_minimize.success

	x = z0[:n_basic_parameters]

	return x, f


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

		np.save('G_old.npy', A_ub)
		np.save('h_old.npy', b_ub)
		np.save('c_old.npy', c)

		import ipdb; ipdb.set_trace()

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
