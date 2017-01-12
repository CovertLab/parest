
from __future__ import division

# from itertools import izip

# from os.path import join

from collections import namedtuple

import numpy as np
# import matplotlib.pyplot as plt

# import liveout as lo

# from utils.bounds import unbounded_to_random
# import utils.linalg as la

# import system
# import fitting
# import th_system

def basic_normal_perturbation(
		initial_deviation,
		final_deviation,
		max_iterations
		):

	def perturbation_function(optimization_result):
		old_x = optimization_result.x

		index = np.random.randint(old_x.size)

		new_x = old_x.copy()

		deviation = initial_deviation * (final_deviation/initial_deviation)**(
				optimization_result.i / max_iterations
				)

		new_x[index] += deviation * np.random.normal()

		return new_x

	return perturbation_function

def optimize(
		x0,
		objective_function,
		perturbation_function,
		):

	x = x0.copy()
	objective_value = objective_function(x0)
	iteration = 0

	best_result = OptimizationResult(x, objective_value, iteration)

	iterate = OptimizationIterate(
		iteration,
		best_result,
		best_result,
		True
		)

	yield iterate

	while True:
		iteration += 1

		test_x = perturbation_function(best_result)
		test_objective_value = objective_function(test_x)

		test_result = OptimizationResult(
			test_x,
			test_objective_value,
			iteration
			)

		did_accept = (test_result.f < best_result.f)

		if did_accept:
			best_result = test_result

		else:
			best_result = OptimizationResult(
				best_result.x,
				best_result.f,
				iteration
				)

		iterate = OptimizationIterate(
			iteration,
			best_result,
			test_result,
			did_accept
			)

		yield iterate

OptimizationIterate = namedtuple(
	'OptimizationIterate',
	('iteration', 'best', 'test', 'did_accept')
	)

OptimizationResult = namedtuple(
	'OptimizationResult',
	('x', 'f', 'i')
	)

if __name__ == '__main__':
	MAX_ITERATIONS = int(1e5)
	INIT_DEV = 1e1
	FINAL_DEV = 1e-5
	N_DIM = 10

	x0 = np.random.normal(size = N_DIM)
	objective_function = lambda x: np.sum(np.square(x))
	perturbation_function = basic_normal_perturbation(
		INIT_DEV,
		FINAL_DEV,
		MAX_ITERATIONS
		)

	history = []

	for iterate in optimize(
			x0,
			objective_function,
			perturbation_function
			):

		if iterate.did_accept:
			history.append(iterate.best)

		if iterate.iteration > MAX_ITERATIONS:
			break

	import matplotlib.pyplot as plt

	plt.figure()

	# plt.subplot(1, 2, 1)
	plt.semilogy(
		[result.i for result in history],
		[result.f for result in history],
		'k.'
		)

	# plt.subplot(1, 2, 2)
	# plt.plot(
	# 	[result.x[0] for result in history],
	# 	[result.x[1] for result in history],
	# 	'k.-'
	# 	)

	plt.show()

