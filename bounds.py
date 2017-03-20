
from __future__ import division

import numpy as np

import constants
import structure

BOUNDS_MATRIX = structure.activity_matrix
INVERSE_BOUNDS_MATRIX = np.linalg.pinv(BOUNDS_MATRIX)

BOUNDS_SATURATED_REACTION_POTENTIAL = (
	-constants.RT * np.log(1e3/constants.K_STAR),
	-constants.RT * np.log(1e-12/constants.K_STAR),
	)

BOUNDS_BINDING_POTENTIAL = (
	-constants.RT * np.log(1e15),
	-constants.RT * np.log(1e-15),
	)

BOUNDS_GIBBS_LOG_CONC = (
	constants.RT * np.log(1e-12),
	constants.RT * np.log(1e3),
	)

(LOWERBOUNDS, UPPERBOUNDS) = np.column_stack(
	[BOUNDS_SATURATED_REACTION_POTENTIAL] * (
		structure.forward_saturated_reaction_potential_matrix.shape[0]
		+ structure.reverse_saturated_reaction_potential_matrix.shape[0]
		)
	+ [BOUNDS_BINDING_POTENTIAL] * (
		structure.solo_forward_binding_potential_matrix.shape[0]
		+ structure.solo_reverse_binding_potential_matrix.shape[0]
		)
	+ [BOUNDS_GIBBS_LOG_CONC] * structure.glc_association_matrix.shape[0]
	)
