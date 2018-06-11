
'''

Deifnes a plotting utility function.  Unused but helpful for inspecting matrices.

'''

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

def matshow(x):
	m = max(x.max(), -x.min())

	plt.imshow(
		x,
		vmin = -m, vmax = +m,
		cmap = 'RdBu',
		interpolation = 'nearest'
		)
