
'''
Plotting code for drawing the condensed box-plots for figures 5 and 6.
'''

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import constants

BLACK = (0.0,)*3
MEDIUM_GRAY = (0.6,)*3
LIGHT_GRAY = (0.8,)*3

TWOFOLD = constants.RT * np.log(2)
TENFOLD = constants.RT * np.log(10)
BILFOLD = constants.RT * np.log(1e9)

DPI = 1000

TENFOLD_STYLE = dict(color = MEDIUM_GRAY, lw = 0.5, zorder = -2)
ONEFOLD_STYLE = dict(color = LIGHT_GRAY, lw = 0.5, zorder = -2)

TYPE_DIVIDER_STYLE = dict(color = LIGHT_GRAY, lw = 0.5, zorder = -1)
RANGE_STYLE = dict(c = BLACK, lw = 0.5)
IQR_STYLE = dict(c = BLACK, lw = 2)
MIDPOINT_STYLE = (
	dict(
		marker = '|',
		c = BLACK,
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

WELL_FIT_COLOR = np.array((78, 105, 177), np.float64)/255.

POORLY_FIT_COLOR = np.array((219, 29, 61), np.float64)/255.

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

from collections import namedtuple

ResidaulsStatistics = namedtuple(
	'ResidaulsStatistics',
	(
		'n_unique',
		'n_within_2x_median',
		'n_within_10x_median',
		'n_within_iqr',
		'n_within_range',
		'n_with_narrow_range',
		'n_with_very_narrow_range'
		)
	)

def plot(residuals, indexing, return_stats = False):
	medians = np.median(residuals, 1)

	(unique, unique_indices, inverse) = np.unique(
		indexing,
		return_index = True,
		return_inverse = True
		)

	n_unique = unique.size

	fig = plt.figure(
		figsize = (1, 6/(47)*(n_unique)), # weird calculation but it has nice dimensions
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

	n_within_2x_median = 0
	n_within_10x_median = 0
	n_within_iqr = 0
	n_within_range = 0
	n_with_narrow_range = 0
	n_with_very_narrow_range = 0

	for i in xrange(n_unique):
		observation_indexes = np.where(inverse == i)[0]

		observations_by_prediction.append(
			-(medians[observation_indexes] - medians[unique_indices[i]])
			- medians[unique_indices[i]]
			)

	for i in xrange(n_unique):
		obs_median = np.median(observations_by_prediction[i])

		if np.abs(obs_median) < TWOFOLD:
			n_within_2x_median += 1

		if np.abs(obs_median) < TENFOLD:
			n_within_10x_median += 1

		if prediction_ranges[i]['lower'] < obs_median < prediction_ranges[i]['upper']:
			n_within_iqr += 1

		if prediction_ranges[i]['smallest'] < obs_median < prediction_ranges[i]['largest']:
			n_within_range += 1

		if prediction_ranges[i]['largest'] - prediction_ranges[i]['smallest'] < TENFOLD:
			n_with_narrow_range += 1

		if prediction_ranges[i]['largest'] - prediction_ranges[i]['smallest'] < TWOFOLD:
			n_with_very_narrow_range += 1

	stats = ResidaulsStatistics(
		n_unique,
		n_within_2x_median,
		n_within_10x_median,
		n_within_iqr,
		n_within_range,
		n_with_narrow_range,
		n_with_very_narrow_range,
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

		axes.axhline(-i + 0.3125, lw = 1.0, c = 'w')

	axes.axhline(-n_unique + 0.3125, lw = 1.0, c = 'w')

	axes.set_ylim(-n_unique + 0.3125, +0.3125)
	axes.set_xlim(-BILFOLD, +BILFOLD)

	axes.axis('off')

	if return_stats:
		return fig, stats

	else:
		return fig
