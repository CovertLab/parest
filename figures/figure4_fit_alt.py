
import os

import figures.figure4_fit

inputs_and_outputs = tuple(
	(
		os.path.join('out', 'all_scaled_upper_sat_limits_{}'.format(penalty)),
		os.path.join('figure4', 'saturation penalized', 'penalty_{}'.format(penalty))
		)
	for penalty in ('1e-1', '1e0', '1e1', '1e2')
	)

for input_and_output in inputs_and_outputs:
	figures.figure4_fit.main(*input_and_output)
	print ''
