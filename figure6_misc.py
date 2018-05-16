
import os

import figure6

for penalty in ('1e-1', '1e0', '1e1', '1e2'):
	figure6.main(
		os.path.join('out', 'all_scaled_upper_sat_limits_{}'.format(penalty)),
		os.path.join('figure6', 'saturation penalized', 'penalty_{}'.format(penalty))
		)
