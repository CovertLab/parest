
from __future__ import division

import numpy as np
from scipy.optimize import minimize, linprog

from utils.linalg import nullspace_projector

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

def build_initial_parameter_values( # TODO: meaningful defaults
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

	# TODO: separate upper and lower bounds for a better interface
	# TODO: same for lower penalty thresholds

	A_bounds = np.column_stack([
		upper_bounds_matrix,
		np.zeros((n_basic_bounds, n_free_parameters))
		])
	h_bounds = upper_bounds_values

	A_absolute_penalty_pos = np.column_stack([
		+absolute_penalty_matrix,
		np.zeros((n_absolute_penalties, n_free_parameters)),
		])
	h_absolute_penalty_pos = +absolute_penalty_values

	A_absolute_penalty_neg = -A_absolute_penalty_pos
	h_absolute_penalty_neg = -h_absolute_penalty_pos

	A_upper_penalty_pos = np.column_stack([
		+upper_penalty_matrix,
		np.zeros((n_upper_penalties, n_free_parameters))
		])
	h_upper_penalty_pos = upper_penalty_values

	 # exceptional case because penalty is not symmetric
	A_upper_penalty_neg = np.zeros_like(A_upper_penalty_pos)
	h_upper_penalty_neg = np.zeros_like(h_upper_penalty_pos)

	if (n_relative_penalty_sets == 0):
		A_relative_penalty_pos = np.zeros((0, n_basic_parameters + n_free_parameters))
		h_relative_penalty_pos = np.zeros((0,))

	else:
		A_relative_penalty_pos = np.row_stack([np.column_stack([
				+relative_penalty_matrix,
				-column_of_ones_matrix(i, relative_penalty_matrix.shape[0], n_relative_penalty_sets),
				])
			for i, (relative_penalty_matrix, relative_penalty_values)
			in enumerate(relative_penalty_matrices_and_values)
			])
		h_relative_penalty_pos = np.concatenate([
			relative_penalty_values
			for relative_penalty_matrix, relative_penalty_values
			in relative_penalty_matrices_and_values
			])

	A_relative_penalty_neg = -A_relative_penalty_pos
	h_relative_penalty_neg = -h_relative_penalty_pos

	A_pos = np.row_stack([
		A_absolute_penalty_pos,
		A_upper_penalty_pos,
		A_relative_penalty_pos,
		])
	A_neg = np.row_stack([
		A_absolute_penalty_neg,
		A_upper_penalty_neg,
		A_relative_penalty_neg,
		])

	I = np.identity(n_hidden_parameters)

	G_bounds = np.column_stack([A_bounds, np.zeros((n_basic_bounds, n_hidden_parameters))])

	G_pos = np.column_stack([A_pos, -I])
	h_pos = np.concatenate([
		h_absolute_penalty_pos,
		h_upper_penalty_pos,
		h_relative_penalty_pos,
		])

	G_neg = np.column_stack([A_neg, -I])
	h_neg = np.concatenate([
		h_absolute_penalty_neg,
		h_upper_penalty_neg,
		h_relative_penalty_neg,
		])

	G = np.row_stack([
		G_bounds,
		G_pos,
		G_neg
		])
	h = np.concatenate([
		h_bounds,
		h_pos,
		h_neg
		])

	c = np.concatenate([
		np.zeros(n_basic_parameters + n_free_parameters),
		np.ones(n_hidden_parameters)
		])

	result_stage1 = linprog(
		c,
		G, h,
		bounds = (None, None),
		options = dict(
			tol = 1e-6 # default tolerance (1e-15?) is very easy to break - underlying linprog precision issue?
			)
		)

	z0 = result_stage1.x
	x0_aug = z0[:n_basic_parameters+n_free_parameters]

	f = result_stage1.fun

	assert result_stage1.success

	# Step 2: regularize the choice of x in the nullspace of the solution for the LP

	# The goal is to find a specific solution (or a more specific solution) to
	# the initialization problem found in step 1.  Linear programming,
	# particularly with the simplex algorithm, tends to find solutions in the
	# corners of the solution space, where parmeter values are at their
	# extremes.  However, we're generally interested in the middle of the
	# bounded space, so I use a second stage optimization to push any free
	# variables (those in the nullspace of the initialization problem) towards
	# a more `regularized' solution.

	# Arguably this regularization could be accomplished by a weakly weighted
	# term in the fitting problem, but I'd need to search for the critical
	# weight.  I also logically prefer the second-order penalty used in this
	# constrained quadratic program.

	is_positive = A_upper_penalty_pos.dot(x0_aug) > h_upper_penalty_pos

	A_active = np.row_stack([
		A_absolute_penalty_pos,
		A_upper_penalty_pos[is_positive, :],
		A_relative_penalty_pos,
		])

	if A_active.size == 0:
		N = np.identity(n_basic_parameters+n_free_parameters)

	else:
		N = nullspace_projector(A_active)

	A_stage2 = np.column_stack([
			init_matrix,
			np.zeros((init_values.size, n_free_parameters))
			])
	b_stage2 = init_values - A_stage2.dot(x0_aug)
	AN_stage2 = A_stage2.dot(N)

	G_stage2 = np.row_stack([
		A_bounds,
		A_upper_penalty_pos[~is_positive, :]
		])
	h_stage2 = np.concatenate([
		h_bounds,
		h_upper_penalty_pos[~is_positive]
		]) - G_stage2.dot(x0_aug)
	GN_stage2 = G_stage2.dot(N)

	result_stage2 = minimize(
		lambda dx_aug: np.sum(np.square(
			AN_stage2.dot(dx_aug) - b_stage2
			)),
		np.zeros_like(x0_aug),
		constraints = dict(
			type = 'ineq',
			fun = lambda dx_aug: h_stage2 - GN_stage2.dot(dx_aug)
			)
		)

	x_aug = x0_aug + N.dot(result_stage2.x)
	x = x_aug[:n_basic_parameters]

	assert result_stage2.success

	# TODO: error checking on stage 1, 2 output

	return x, f

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
