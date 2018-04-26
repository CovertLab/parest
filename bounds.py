
from __future__ import division

import numpy as np

import constants
import structure

'''
The bounds of our new, transformed system (parsimonious perturbations).  Bounds
the lower and upper values of:
- concentrations of dynamic species
- maximum forward and reverse rates of reaction
- saturation ratios
'''

BOUNDS_MATRIX = structure.activity_matrix
INVERSE_BOUNDS_MATRIX = np.linalg.pinv(BOUNDS_MATRIX)

RESOLUTION = np.finfo(np.float64).resolution # "resolution" of 64-bit floating point number, = 1e-15

'''
Concentration are bounded from 1 fM to 1 MM.  Both are more permissive than
expected values.  A concentration of one molecule per E. coli cell is roughly
1 nM, while water, the most abundant species, has a concentration of about
50 M.  The range is consistent with the RESOLUTION.
'''
LOWER_CONC = 1e-12
UPPER_CONC = 1e3

'''
Choosing upper and lower v_max values is difficult, however we know from
fluxomics that an active pathway (like glycolysis) operates around a net
flux of 0.1 mM/s.  These bounds are centered about an average of about 3 uM/s,
and with a range consistent with the RESOLUTION.  The hope is that this average
is low enough to be representative of the average flux, and far enough from
our target glycolysis flux to not bias initialization.  At the same time, a
flux that is too small could be tough for the system to escape.

Note that this is just v_max, not a total constraint on v.  Assuming enzymes
that are at least somewhat unsaturated, the real "average" v's should be even
smaller.
'''
LOWER_VMAX = 1e-13
UPPER_VMAX = 1e2

BOUNDS_SATURATED_REACTION_POTENTIAL = (
	-constants.RT * np.log(UPPER_VMAX/constants.K_STAR),
	-constants.RT * np.log(LOWER_VMAX/constants.K_STAR),
	)

'''
The upper and lower bounds on binding ratios were selected with the
knowledge that all binding ratios are relative to the "unbound" ratio, which
in our generalized kinetic model is 1.  Thus, for numerical significance, the
minimum saturation ratio is RESOLUTION and the maximum is 1/RESOLUTION. The
range here is quite wide but that was needed to accomadate the range in the
training data.
'''
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
