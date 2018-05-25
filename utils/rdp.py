
# TODO: delete?

from __future__ import division

import numpy as np

def _squared_orthogonal_distance(first_point, last_point, points):
	last_point = last_point - first_point
	points = points - first_point

	return (
		np.sum(points * points, 1)
		- np.square(
			np.sum(points * last_point, 1)
			) / np.sum(last_point * last_point)
		)

def rdp(points, threshold):
	# each row in 'points' is the coordinates of a point

	n_points = points.shape[0]

	squared_threshold = np.square(threshold)

	keep = np.zeros(n_points, np.bool)
	active = np.ones(n_points, np.bool)

	keep[0] = True
	keep[-1] = True

	active[0] = False
	active[-1] = False

	first = 0

	while np.any(active):

		first = np.where(~active[first:-1] & active[first+1:])[0][0] + first
		last = np.where(~active[(first+1):])[0][0] + (first+1)

		between = slice((first+1), last)

		square_distances = _squared_orthogonal_distance(
			points[first, :],
			points[last, :],
			points[between, :]
			)

		largest = np.argmax(square_distances)

		if square_distances[largest] > squared_threshold:
			keep[largest + (first+1)] = True
			active[largest + (first+1)] = False

		else:
			keep[between] = False
			active[between] = False

	return keep


if __name__ == '__main__':
	raise Exception('This demonstration code is broken without installing the third-party RDP package.')

	from time import time

	import matplotlib.pyplot as plt

	from rdp import rdp as rdp_slow

	ERROR = 1e-2
	LINTHRESH = 1e-3
	N = int(1e4)

	base10_sinh = lambda x: (10**x - 10**-x)/2
	base10_arcsinh = lambda x: np.log10(x + np.sqrt(np.square(x)+1))

	test_x = np.linspace(0, 10, N)
	test_y = np.sin(test_x * np.pi) * np.exp(-test_x/2)

	points = np.column_stack([test_x, base10_arcsinh(test_y/LINTHRESH)])
	# points = np.column_stack([test_x, test_y])

	# Slow method

	t0 = time()
	slow_x, arcsinh_slow_y = rdp_slow(points, ERROR).T
	slow_y = base10_sinh(arcsinh_slow_y)*LINTHRESH
	# slow_x, slow_y = rdp_slow(points, ERROR).T
	t_slow = time() - t0

	plt.subplot(1, 2, 1)
	plt.plot(test_x, test_y, '-', lw = 9, color = (0.8,)*3)
	plt.plot(slow_x, slow_y, '--.', lw = 3, ms = 15, color = (0.2,)*3)

	plt.title(
		'Using rdp pacakge\n{} of {} ({:.2%}) points retained\n{:.2e} seconds to evaluate'.format(
			slow_x.size, test_x.size, slow_x.size/test_x.size, t_slow
			))

	plt.xlim(-0.5, 10.5)
	plt.ylim(-0.6, +0.9)

	plt.yscale('symlog', linthreshy = LINTHRESH)

	# Custom method

	t0 = time()
	mask = rdp(points, ERROR)
	t_fast = time() - t0

	reduced_x = test_x[mask]
	reduced_y = test_y[mask]

	plt.subplot(1, 2, 2)
	plt.plot(test_x, test_y, '-', lw = 9, color = (0.8,)*3)
	plt.plot(reduced_x, reduced_y, '--.', lw = 3, ms = 15, color = (0.2,)*3)

	plt.title(
		'Using custom RDP implementation\n{} of {} ({:.2%}) points retained\n{:.2e} seconds to evaluate ({:.2e} times faster)'.format(
			mask.sum(), mask.size, mask.sum()/mask.size, t_fast, t_slow/t_fast
			))

	plt.xlim(-0.5, 10.5)
	plt.ylim(-0.6, +0.9)

	plt.yscale('symlog', linthreshy = LINTHRESH)

	plt.show()
