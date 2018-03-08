
from __future__ import division

import numpy as np

import problems

import fitting

fitting_rules_and_weights = problems.DEFINITIONS['all_scaled']

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

print fitting_values.size + upper_fitting_values.size + sum(
	rfts[1].size
	for rfts in relative_fitting_tensor_sets
	)
