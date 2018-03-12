
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt

class SoftMarginSVM(object):
	"""
	Performs soft-margin SVM using gradient descent.  Returns an object
	with a pair of attributes defining the optimal separating
	hyperplane.

	Initialization arguments
	------------------------

	points: float-(n x m)-matrix of point positions
	accepted: boolean-n-vector, True for each point that is in the 'accepted' class
	softness: float, small positive weight on minimizing the magnitude
		of the direction vector

	Attributes
	----------

	.direction: m-vector
	.offset: float

	The equation direction.dot(x) = offset defines the hyperplane
	that best separates the 'reject' and 'accept' points.  The margins
	are given by the same formula but with 'offset' increased or
	decreased by 1.

	Metaparameters
	--------------

	These are the parameters associated with the gradient descent
	operation.  Subclass to modify the default metaparameter values.

	max_iterations = 10000: max number of gradient descent iterations
	initial_stepsize = 0.1: size of first gradient descent step
	stepsize_increase = 1.1: multiplier on stepsize following a successful step, should be >1
	stepsize_decrease = 2: divider on stepsize following a failed step, should be >1
	min_stepsize = 1e-3: evaluation stops if the stepsize falls below this threshold, should be >0

	TODO: option for softmax i.e. ln(1 + exp(#)) instead of max(0, #)
	TODO: option for weighting each point
	TODO: abstract out the gradient descent?  might be hard to store intermediate calculations

	"""

	max_iterations = 10000

	initial_stepsize = 0.1

	stepsize_increase = 1.1
	stepsize_decrease = 2.0

	min_stepsize = 1e-3

	def __init__(self, points, accepted, softness):
		classes = np.ones(points.shape[0], np.float64)
		classes[~accepted] = -1.0

		dg_dw = -classes * points.T
		dg_db = classes

		n_inv = 1/classes.size # pre-invert for the mild performance gains

		n_accept = accepted.sum()
		n_reject = (~accepted).sum()

		mean_accept = np.mean(points[accepted], 0)
		mean_reject = np.mean(points[~accepted], 0)

		# TODO: evaluate use of weighted average - unweighted might make more sense
		mean_all = (n_accept * mean_accept + n_reject * mean_reject) / (n_accept + n_reject)

		self.direction = mean_accept - mean_reject
		self.offset = mean_all.dot(self.direction)

		stepsize = self.initial_stepsize

		hinge_loss = self._calc_hinge_loss(classes, points, self.direction, self.offset)

		obj = self._calc_objective(n_inv, hinge_loss, softness, self.direction)

		(grad_obj_w, grad_obj_b) = self._calc_grad(dg_dw, dg_db, hinge_loss, n_inv, softness, self.direction)

		for self.iteration in xrange(self.max_iterations):
			new_direction = self.direction - stepsize * grad_obj_w
			new_offset = self.offset - stepsize * grad_obj_b

			new_hinge_loss = self._calc_hinge_loss(classes, points, new_direction, new_offset)

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

	# TODO: separate class that handles these calculations?

	@staticmethod
	def _calc_hinge_loss(classes, points, direction, offset):
		"""
		Computes the hinge loss for the soft-margin SVM problem.  The soft
		margin SVM penalizes points on the wrong side of the hyperplane
		linearly with respect to distance from the hyperplane.

		TODO: consider using precomputed classes * points
		"""
		return np.fmax(0, 1 - classes * (points.dot(direction) - offset).T)

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

		return (grad_obj_w, grad_obj_b)

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	N_REJECT = 300
	N_ACCEPT = 300

	np.random.seed(83)

	def plot_class(points, color):
		(x, y) = points.T

		plt.plot(x, y, '.', color = color, ms = 3)

	reject = np.random.normal(size = (N_REJECT, 2)) - 1
	accept = np.random.normal(size = (N_ACCEPT, 2)) + 3

	points = np.concatenate([reject, accept])
	accepted = np.ones(N_REJECT + N_ACCEPT, np.bool)
	accepted[:N_REJECT] = False

	softness = 1e-2

	result = SoftMarginSVM(points, accepted, softness)

	plt.figure(figsize = (6, 6))

	plot_class(reject, 'r')
	plot_class(accept, 'b')

	# Draw separating hyperplane and margins
	# TODO: ensure that ends exceed plotted region

	x = np.linspace(-100, +100, 2)

	plt.plot(
		x, (result.offset - result.direction[0] * x) / result.direction[1],
		'k-',
		lw = 1.5
		)

	plt.plot(
		x, (result.offset+1 - result.direction[0] * x) / result.direction[1],
		'k--',
		lw = 0.5
		)

	plt.plot(
		x, (result.offset-1 - result.direction[0] * x) / result.direction[1],
		'k--',
		lw = 0.5
		)

	plt.xlim(-4, +6)
	plt.ylim(-4, +6)

	plt.savefig('svm.png')
