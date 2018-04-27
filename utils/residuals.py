
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import constants

DARK_GRAY = (0.0,)*3
MEDIUM_GRAY = (0.5,)*3
LIGHT_GRAY = (0.8,)*3

TENFOLD = constants.RT * np.log(10)
BILFOLD = constants.RT * np.log(1e9)

DPI = 1000

TENFOLD_STYLE = dict(color = MEDIUM_GRAY, lw = 0.5, zorder = -2)
ONEFOLD_STYLE = dict(color = LIGHT_GRAY, lw = 0.5, zorder = -2)

TYPE_DIVIDER_STYLE = dict(color = LIGHT_GRAY, lw = 0.5, zorder = -1)
RANGE_STYLE = dict(c = DARK_GRAY, lw = 0.5)
IQR_STYLE = dict(c = DARK_GRAY, lw = 2)
MIDPOINT_STYLE = (
	dict(
		marker = '|',
		c = DARK_GRAY,
		ms = 2,
		markeredgewidth = 0.5,
		)
	)

OBSERVATION_STYLE = (
	dict(
		marker = '^',
		ms = 2.5,
		markeredgewidth = 0.1,
		markeredgecolor = 'w'
		)
	)

OBSERVATION_OFFSET = 0.35

from matplotlib.cm import RdBu

COLOR_OFFSET = 0.1

WELL_FIT_COLOR = (
	# (0.1, 0.3, 1,)
	# RdBu(1.0 - COLOR_OFFSET)
	'royalblue'
	)

POORLY_FIT_COLOR = (
	# (0.9, 0.1, 0.1)
	# RdBu(0.0 + COLOR_OFFSET)
	'crimson'
	)

WELL_FIT_OBSERVATION_STYLE = OBSERVATION_STYLE.copy()
WELL_FIT_OBSERVATION_STYLE.update(
	c = WELL_FIT_COLOR,
	zorder = 3
	)

POORLY_FIT_OBSERVATION_STYLE = OBSERVATION_STYLE.copy()
POORLY_FIT_OBSERVATION_STYLE.update(
	c = POORLY_FIT_COLOR,
	zorder = 4
	)

def _get_common_value(*values):
	value = values[0]

	assert all(value == v2 for v2 in values), 'multiple values: {}'.format(values)

	return value

def plot(residuals, indexing):
	medians = np.median(residuals, 1)

	(unique, unique_indices, inverse) = np.unique(
		indexing,
		return_index = True,
		return_inverse = True
		)

	n_unique = unique.size

	fig = plt.figure(
		figsize = (1, 6/(45+1)*(n_unique+1)), # weird calculation but it has nice dimensions
		dpi = DPI  # DPI set here doesn't matter much
		)

	prediction_ranges = np.empty(
		n_unique,
		[
			('smallest', np.float64),
			('lower', np.float64),
			('upper', np.float64),
			('largest', np.float64),
			]
		)

	for i, ind in enumerate(unique_indices):
		prediction_ranges[i] = tuple(np.percentile(
			residuals[ind, :],
			(
				0,
				25,
				75,
				100
				)
			) - medians[ind])

	observations_by_prediction = []

	for i in xrange(n_unique):
		observation_indexes = np.where(inverse == i)[0]

		observations_by_prediction.append(
			-(medians[observation_indexes] - medians[unique_indices[i]])
			- medians[unique_indices[i]]
			)

	axes = fig.add_axes((0, 0, 1, 1))

	axes.axvline(0, **ONEFOLD_STYLE)
	axes.axvline(-TENFOLD, **TENFOLD_STYLE)
	axes.axvline(+TENFOLD, **TENFOLD_STYLE)

	for i in xrange(n_unique):
		axes.plot(
			(prediction_ranges[i]['smallest'], prediction_ranges[i]['largest']),
			(-i, -i),
			**RANGE_STYLE
			)

		axes.plot(
			(prediction_ranges[i]['lower'], prediction_ranges[i]['upper']),
			(-i, -i),
			**IQR_STYLE
			)

		axes.plot(0, -i, **MIDPOINT_STYLE)

		for obs in np.sort(observations_by_prediction[i]):
			style = (
				WELL_FIT_OBSERVATION_STYLE
				if np.abs(obs) < TENFOLD
				else POORLY_FIT_OBSERVATION_STYLE
				)
			axes.plot(obs, -i-OBSERVATION_OFFSET, **style)

			if np.abs(obs) > BILFOLD:
				print 'warning - exceeds axis limits'

	axes.set_ylim(-n_unique+0.5, +0.5)
	axes.set_xlim(-BILFOLD, +BILFOLD)

	axes.axis('off')

	return fig
