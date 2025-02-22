
from __future__ import division

import numpy as np

import problems

import fitting

from utils.linalg import nullspace_projector

import structure

def get_determinacy_vectors(problem):
	fitting_rules_and_weights = problems.DEFINITIONS[problem]

	fitting_tensors = (
			fitting_matrix,
			fitting_values,
			fitting_entries
			) = fitting.build_fitting_tensors(*fitting_rules_and_weights)

	upper_fitting_tensors = (
		upper_fitting_matrix,
		upper_fitting_values,
		upper_fitting_entries
		) = fitting.build_upper_fitting_tensors(*fitting_rules_and_weights)

	relative_fitting_tensor_sets = fitting.build_relative_fitting_tensor_sets(
		*fitting_rules_and_weights
		)

	n_original_variables = fitting_matrix.shape[1]
	n_accessory_variables = len(relative_fitting_tensor_sets)

	def ones_column_matrix(n, m, i):
		out = np.zeros((n, m))
		out[:, i] = 1
		return out

	full_fitting_matrix = np.vstack([
		np.hstack([fitting_matrix, np.zeros((fitting_matrix.shape[0], n_accessory_variables))]),
		np.hstack([upper_fitting_matrix, np.zeros((upper_fitting_matrix.shape[0], n_accessory_variables))]),
		] + [
		np.hstack([rfts[0], -ones_column_matrix(rfts[0].shape[0], n_accessory_variables, i)])
		for i, rfts in enumerate(relative_fitting_tensor_sets)
		])

	undetermined = np.all(full_fitting_matrix == 0, 0)[:n_original_variables]

	nullspace = np.round(nullspace_projector(full_fitting_matrix), 10) # 64bit floating point precision is about 15 decimal places; I go with 10 to be safe

	fully_determined = np.all(nullspace == 0, 0)[:n_original_variables]

	partially_determined = ~undetermined & ~fully_determined

	return undetermined, partially_determined, fully_determined

u1, p1, f1 = get_determinacy_vectors('all_scaled')

u2, p2, f2 = get_determinacy_vectors('all_scaled_upper_sat_limits_1e0')

print (u1 & u2).sum(), (p1 & u2).sum(), (f1 & u2).sum(), u2.sum()
print (u1 & p2).sum(), (p1 & p2).sum(), (f1 & p2).sum(), p2.sum()
print (u1 & f2).sum(), (p1 & f2).sum(), (f1 & f2).sum(), f2.sum()
print u1.sum(), p1.sum(), f1.sum(), structure.n_parameters
