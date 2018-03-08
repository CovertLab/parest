
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt

def _calc_hinge_loss(classes, all_points, direction, offset):
	return 1 - classes * (all_points.dot(direction) - offset).T

def _calc_objective(n_inv, h, softness, direction):
	return n_inv * np.sum(np.fmax(0, h)) + softness * direction.dot(direction)

def _calc_grad(classes, all_points, h, n_inv, softness, direction):
	dg_dw = - classes * all_points.T
	dg_db = classes

	h_gtz = (h > 0)

	grad_obj_w = n_inv * np.sum(dg_dw[:, h_gtz], 1) + 2 * softness * direction
	grad_obj_b = n_inv * np.sum(dg_db[h_gtz])

	return grad_obj_w, grad_obj_b


def svm(
		points_reject, points_accept,
		softness,
		max_iterations = 10000,
		):
	"""

	Inputs
	------

	points_reject: (n1 x m)-matrix of point positions
	points_accept: (n2 x m)-matrix of point positions
	softness: float, small positive weight on minimizing the magnitude
		of the direction vector

	Outputs
	-------

	direction: m-vector
	offset: float

	direction.dot(x) = offset defines the separating hyperplane

	TODO: more optional arguments for optimization metaparameters
	TODO: option for softmax i.e. ln(1 + exp(#)) instead of max(0, #)

	"""

	all_points = np.concatenate([points_reject, points_accept])

	classes = np.concatenate([
		-np.ones(points_reject.shape[0]),
		+np.ones(points_accept.shape[0]),
		])

	n_inv = 1/classes.size

	direction = np.random.normal(size = all_points.shape[1])
	offset = 0

	stepsize = 0.1

	h = _calc_hinge_loss(classes, all_points, direction, offset)

	obj = _calc_objective(n_inv, h, softness, direction)

	(grad_obj_w, grad_obj_b) = _calc_grad(classes, all_points, h, n_inv, softness, direction)

	for iteration in xrange(max_iterations):
		new_direction = direction - stepsize * grad_obj_w
		new_offset = offset - stepsize * grad_obj_b

		new_h = _calc_hinge_loss(classes, all_points, new_direction, new_offset)

		new_obj = _calc_objective(n_inv, new_h, softness, new_direction)

		if new_obj < obj:
			direction = new_direction
			offset = new_offset

			h = new_h
			obj = new_obj

			(grad_obj_w, grad_obj_b) = _calc_grad(classes, all_points, h, n_inv, softness, direction)

			stepsize *= 1.1

		else:
			stepsize /= 2

		if stepsize <= 1e-3:
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
