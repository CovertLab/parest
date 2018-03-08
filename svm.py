
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt


def _calc_hinge_loss(classes, all_points, direction, offset):
	"""
	Computes the hinge loss for the soft-margin SVM problem.  The soft
	margin SVM penalizes points on the wrong side of the hyperplane
	linearly with respect to distance from the hyperplane.

	See 'svm' function for descriptions of arguments.

	TODO: consider using precomputed classes * all_points
	"""
	return np.fmax(0, 1 - classes * (all_points.dot(direction) - offset).T)


def _calc_objective(n_inv, hinge_loss, softness, direction):
	"""
	Calculates the objective value for the soft-margin SVM problem.
	Makes use of the pre-computed hinge loss since we also use that
	intermediate calculation in the gradient.

	See 'svm' function for descriptions of arguments.
	"""

	return n_inv * np.sum(hinge_loss) + softness * direction.dot(direction)


def _calc_grad(dg_dw, dg_db, hinge_loss, n_inv, softness, direction):
	"""
	Calculates the gradient on the direction vector ('w') and the offset
	value ('b').  Makes use of the pre-computed hinge loss since we
	also use that intermediate calculation in the objective.

	See 'svm' function for descriptions of arguments.
	"""
	bad_point = (hinge_loss > 0) # points on the wrong side of the separating hyperplane

	grad_obj_w = n_inv * np.sum(dg_dw[:, bad_point], 1) + 2 * softness * direction
	grad_obj_b = n_inv * np.sum(dg_db[bad_point])

	return grad_obj_w, grad_obj_b


def svm(
		points_reject, points_accept,
		softness,
		max_iterations = 10000,
		initial_stepsize = 0.1,
		stepsize_increase = 1.1,
		stepsize_decrease = 2,
		min_stepsize = 1e-3,
		):
	"""

	Required arguments
	------------------

	points_reject: (n1 x m)-matrix of point positions
	points_accept: (n2 x m)-matrix of point positions
	softness: float, small positive weight on minimizing the magnitude
		of the direction vector

	Optional arguments
	------------------

	max_iterations = 10000: max number of gradient descent iterates
	initial_stepsize = 0.1: size of first gradient descent step
	stepsize_increase = 1.1: multiplier on stepsize following a successful step, should be >1
	stepsize_decrease = 2: divider on stepsize following a failed step, should be >1
	min_stepsize = 1e-3: evaluation stops if the stepsize falls below this threshold, should be >0

	Outputs
	-------

	direction: m-vector
	offset: float

	The equation direction.dot(x) = offset defines the hyperplane
	that best separates the 'reject' and 'accept' points.

	TODO: option for softmax i.e. ln(1 + exp(#)) instead of max(0, #)
	TODO: option for weighting
	TODO: implement as a class?

	"""

	all_points = np.concatenate([points_reject, points_accept])

	classes = np.concatenate([
		-np.ones(points_reject.shape[0]),
		+np.ones(points_accept.shape[0]),
		]).astype(np.float64) # int-float multiplication will be recast to float-float

	dg_dw = -classes * all_points.T
	dg_db = classes

	n_inv = 1/classes.size # pre-invert for the mild performance gains

	direction = np.random.normal(size = all_points.shape[1])
	offset = 0

	stepsize = initial_stepsize

	hinge_loss = _calc_hinge_loss(classes, all_points, direction, offset)

	obj = _calc_objective(n_inv, hinge_loss, softness, direction)

	(grad_obj_w, grad_obj_b) = _calc_grad(dg_dw, dg_db, hinge_loss, n_inv, softness, direction)

	for iteration in xrange(max_iterations):
		new_direction = direction - stepsize * grad_obj_w
		new_offset = offset - stepsize * grad_obj_b

		new_hinge_loss = _calc_hinge_loss(classes, all_points, new_direction, new_offset)

		new_obj = _calc_objective(n_inv, new_hinge_loss, softness, new_direction)

		if new_obj < obj:
			direction = new_direction
			offset = new_offset

			hinge_loss = new_hinge_loss
			obj = new_obj

			(grad_obj_w, grad_obj_b) = _calc_grad(dg_dw, dg_db, hinge_loss, n_inv, softness, direction)

			stepsize *= stepsize_increase

		else:
			stepsize /= stepsize_decrease

		if stepsize <= min_stepsize:
			break

	return direction, offset

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	def plot_class(points, color):
		(x, y) = points.T

		plt.plot(x, y, '.', color = color)

	reject = np.random.normal(size = (300, 2)) - 1
	accept = np.random.normal(size = (300, 2)) + 1

	softness = 1e-1

	(direction, offset) = svm(reject, accept, softness)

	plt.figure(figsize = (6, 6))

	plot_class(reject, 'r')
	plot_class(accept, 'b')

	x = np.linspace(-10, +10, 2)

	plt.plot(
		x, (offset - direction[0] * x) / direction[1],
		'k-',
		lw = 3
		)

	plt.xlim(-5, +5)
	plt.ylim(-5, +5)

	plt.savefig('svm.png')
