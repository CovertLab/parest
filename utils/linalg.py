
from __future__ import division

import numpy as np

_RES = np.finfo(np.float).resolution

class CachedLeastSquares(object):
	def __init__(self, matrix, rcond = _RES):
		u, s, vT = np.linalg.svd(matrix, full_matrices = False)

		uinv = np.asarray(u.T, order = 'C')
		vTinv = np.asarray(vT.T, order = 'C')
		sinv = 1/s

		keep = (s > rcond)

		self.uinv = uinv[keep, :]
		self.sinv = np.diag(sinv[keep])
		self.vTinv = vTinv[:, keep]

	def __call__(self, vector):
		return self.vTinv.dot(
			self.sinv.dot(
				self.uinv.dot(vector)
				)
			)

def nullspace_basis(matrix, rcond = _RES):
	u, s, vT = np.linalg.svd(matrix, full_matrices = True)

	try:
		first = np.where(s < rcond)[0][0]

	except IndexError:
		first = s.size

	null = vT[first:, :].T

	return null

def pinv(matrix, rcond = _RES):
	u, s, vT = np.linalg.svd(matrix, full_matrices = False)

	retain = (s / s[0] > rcond)

	u_inv = u[:, retain].T
	s_inv = 1/s[retain]
	vT_inv = vT[retain, :].T

	return vT_inv.dot(np.diag(s_inv).dot(u_inv))

def projector(matrix, rcond = _RES):
	u, s, vT = np.linalg.svd(matrix, full_matrices = True)

	matrix_pseudoinverse = pinv(matrix, rcond)

	return matrix_pseudoinverse.dot(matrix)

def nullspace_projector(matrix, rcond = _RES):
	p = projector(matrix, rcond)

	return np.identity(p.shape[0]) - p

def bilevel_pseudoinverse(first_matrix, second_matrix, rcond = _RES):
	first_matrix_pseudoinverse = pinv(first_matrix, rcond)

	size = first_matrix.shape[1]

	first_nullspace_projector = np.identity(size) - first_matrix_pseudoinverse.dot(first_matrix)

	second_nullspace_projector = np.identity(size) - first_nullspace_projector.dot(
		pinv(second_matrix.dot(first_nullspace_projector), rcond).dot(
			second_matrix
			)
		)

	return second_nullspace_projector.dot(first_matrix_pseudoinverse)

def bilevel_elementwise_pseudoinverse(first_matrix, second_matrix, rcond = _RES):
	n, size = first_matrix.shape

	vectors = [first_matrix[[i], :] for i in xrange(n)]

	bilevel_pinvs = []

	for vector in vectors:
		vp = pinv(vector, rcond)

		first_nullspace_projector = np.identity(size) - vp.dot(vector)

		second_nullspace_projector = np.identity(size) - first_nullspace_projector.dot(
			pinv(second_matrix.dot(first_nullspace_projector), rcond).dot(
				second_matrix
				)
			)

		bilevel_pinvs.append(
			second_nullspace_projector.dot(vp)
			)

	return np.concatenate(bilevel_pinvs, 1)

def approx_jac(f, x, d = 1e-3):
	f0 = f(x)

	j = np.empty((f0.size, x.size))

	if np.shape(d) == tuple():
		d = d * np.ones(x.size)

	for i in xrange(x.size):
		x_plus = x.copy()
		x_plus[i] += d[i]

		x_minus = x.copy()
		x_minus[i] -= d[i]

		f_plus = f(x_plus)
		f_minus = f(x_minus)

		df = ((f_plus - f0)/d[i] + (f0 - f_minus)/d[i])/2

		j[:, i] = df

	return j

def bilevel_lstsq(A, y, B, z, rcond = _RES):
	# minimizes ||Bx - z||
	# subject to also minimizing ||Ax - y||

	nA = nullspace_projector(A, rcond)

	Ap = pinv(A, rcond)

	BnAp = pinv(B.dot(nA), rcond)

	x = nA.dot(BnAp.dot(z)) + (np.identity(nA.shape[0]) - nA.dot(BnAp.dot(B))).dot(Ap.dot(y))

	return x
