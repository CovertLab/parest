
from __future__ import division

import numpy as np

import constants
import structure

BOUNDS_MATRIX = structure.activity_matrix
INVERSE_BOUNDS_MATRIX = np.linalg.pinv(BOUNDS_MATRIX)

RESOLUTION = 1e-15

LOWER_CONC = 1e-12
UPPER_CONC = 1e3

LOWER_VMAX = 1e-12
UPPER_VMAX = 1e3

BOUNDS_SATURATED_REACTION_POTENTIAL = (
	-constants.RT * np.log(UPPER_VMAX/constants.K_STAR),
	-constants.RT * np.log(LOWER_VMAX/constants.K_STAR),
	)

BOUNDS_BINDING_POTENTIAL = (
	-constants.RT * np.log(1/RESOLUTION),
	-constants.RT * np.log(RESOLUTION),
	)

BOUNDS_GIBBS_LOG_CONC = (
	constants.RT * np.log(LOWER_CONC),
	constants.RT * np.log(UPPER_CONC),
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
