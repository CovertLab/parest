'''
A number of accelerated vector operations.  These operations avoid a lot of
boilerplate error/type checking intrinsic to default numpy operations; further
they express many common operations in terms of dot products, which can be
accelerated on floating-point arrays using third-party linear algebra packages.

Optimizing these operations is important because they tend to be evaluated in
innner loops (e.g. in evaluating the objective).
'''

from __future__ import division

import numpy as np

# TODO: Evaluate whether keeping certain pieces of memory allocated reduces
# evaluation time.  I suspect that allocating lots of short arrays can be
# costly.

def fast_shortarray_median1d(x):
	'''
	Computes the median of a vector of values.

	Numpy has a median function, but for short vectors (~100 elements) it is
	slower than sorting the vector and then taking the middle element (or, in
	the case of an even number of elements, taking the average of the two
	middle elements).

	TODO: consider caching divmod result
	'''
	(halfsize, odd_number_of_elements) = divmod(x.size, 2)

	# Notes on sorting:
	# - Sorting algorithm (i.e. kind = ...) makes no appreciable difference
	# - Likewise for np.argsort in place of np.sort
	# - As far as I can reason, you need to sort the whole vector to get the
	#	median right (no obvious optimization to replace np.sort/argsort)
	x_sorted = np.sort(x)

	if odd_number_of_elements:
		median = x_sorted[halfsize]

	else:
		median = (x_sorted[halfsize-1] + x_sorted[halfsize])/2.0

	return median

def fast_shortarray_median1d_partition(x):
	'''
	Computes the median of a vector of values.  This is an alternative
	implementation that utilizes np.partition, which appears to be slower for
	very short array but scales better than using np.sort (which performs a
	full sort rather than np.partition's partial support).
	'''
	(halfsize, odd_number_of_elements) = divmod(x.size, 2)

	if odd_number_of_elements:
		median = np.partition(x, halfsize)[halfsize]

	else:
		partitioned = np.partition(x, (halfsize-1, halfsize))
		median = (partitioned[halfsize-1]+partitioned[halfsize])/2.0

	return median

def fast_shortarray_sumabs1d(x):
	'''
	Compute the sum of absolute values of a vector's elements, a.k.a. the
	L1-norm.

	This operation is faster than np.sum(np.abs(x)) and np.linalg.norm(x, 1),
	at least for small vectors.
	'''
	return np.sign(x).dot(x)

def fast_shortarray_sumsq1d(x):
	'''
	Compute the sum of squares of a vector's elements, a.k.a. the squared
	L2-norm.

	This operation is faster than np.sum(np.square(x)) and squaring
	np.linalg.norm(x, 2), at least for short vectors.
	'''
	return x.dot(x)
