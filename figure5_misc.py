
import os

import figure5

for penalty in ('1e-1', '1e0', '1e1', '1e2'):
	figure5.main(
		os.path.join('out', 'all_scaled_upper_sat_limits_{}'.format(penalty)),
		os.path.join('figure5', 'saturation penalized', 'penalty_{}'.format(penalty))
		)
