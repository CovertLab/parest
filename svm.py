
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt

class SoftMarginSVM(object):
	max_iterations = 10000

	initial_stepsize = 0.1

	stepsize_increase = 1.1
	stepsize_decrease = 2.0

	min_stepsize = 1e-3

	def __init__(self, points_reject, points_accept, softness):
		all_points = np.concatenate([points_reject, points_accept])

		classes = np.concatenate([
			-np.ones(points_reject.shape[0]),
			+np.ones(points_accept.shape[0]),
			]).astype(np.float64) # int-float multiplication will be recast to float-float

		dg_dw = -classes * all_points.T
		dg_db = classes

		n_inv = 1/classes.size # pre-invert for the mild performance gains

		self.direction = np.random.normal(size = all_points.shape[1])
		self.offset = 0

		stepsize = self.initial_stepsize

		hinge_loss = self._calc_hinge_loss(classes, all_points, self.direction, self.offset)

		obj = self._calc_objective(n_inv, hinge_loss, softness, self.direction)

		(grad_obj_w, grad_obj_b) = self._calc_grad(dg_dw, dg_db, hinge_loss, n_inv, softness, self.direction)

		for iteration in xrange(self.max_iterations):
			new_direction = self.direction - stepsize * grad_obj_w
			new_offset = self.offset - stepsize * grad_obj_b

			new_hinge_loss = self._calc_hinge_loss(classes, all_points, new_direction, new_offset)

			new_obj = self._calc_objective(n_inv, new_hinge_loss, softness, new_direction)

			if new_obj < obj:
				self.direction = new_direction
				self.offset = new_offset

				hinge_loss = new_hinge_loss
				obj = new_obj

				(grad_obj_w, grad_obj_b) = self._calc_grad(dg_dw, dg_db, hinge_loss, n_inv, softness, self.direction)

				stepsize *= self.stepsize_increase

			else:
				stepsize /= self.stepsize_decrease

			if stepsize <= self.min_stepsize:
				break

	# These are all static methods because we don't want to override the
	# current values before we decide to accept a step.

	# TODO: separate class that handles these calculations

	@staticmethod
	def _calc_hinge_loss(classes, all_points, direction, offset):
		"""
		Computes the hinge loss for the soft-margin SVM problem.  The soft
		margin SVM penalizes points on the wrong side of the hyperplane
		linearly with respect to distance from the hyperplane.

		TODO: consider using precomputed classes * all_points
		"""
		return np.fmax(0, 1 - classes * (all_points.dot(direction) - offset).T)

	@staticmethod
	def _calc_objective(n_inv, hinge_loss, softness, direction):
		"""
		Calculates the objective value for the soft-margin SVM problem.
		Makes use of the pre-computed hinge loss since we also use that
		intermediate calculation in the gradient.
		"""

		return n_inv * np.sum(hinge_loss) + softness * direction.dot(direction)

	@staticmethod
	def _calc_grad(dg_dw, dg_db, hinge_loss, n_inv, softness, direction):
		"""
		Calculates the gradient on the direction vector ('w') and the offset
		value ('b').  Makes use of the pre-computed hinge loss since we
		also use that intermediate calculation in the objective.
		"""
		bad_point = (hinge_loss > 0) # points on the wrong side of the separating hyperplane

		grad_obj_w = n_inv * np.sum(dg_dw[:, bad_point], 1) + 2 * softness * direction
		grad_obj_b = n_inv * np.sum(dg_db[bad_point])

		return grad_obj_w, grad_obj_b

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	def plot_class(points, color):
		(x, y) = points.T

		plt.plot(x, y, '.', color = color)

	reject = np.random.normal(size = (300, 2)) - 1
	accept = np.random.normal(size = (300, 2)) + 1

	softness = 1e-1

	result = SoftMarginSVM(reject, accept, softness)

	plt.figure(figsize = (6, 6))

	plot_class(reject, 'r')
	plot_class(accept, 'b')

	x = np.linspace(-10, +10, 2)

	plt.plot(
		x, (result.offset - result.direction[0] * x) / result.direction[1],
		'k-',
		lw = 3
		)

	plt.xlim(-5, +5)
	plt.ylim(-5, +5)

	plt.savefig('svm.png')
