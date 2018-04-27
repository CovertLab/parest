
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt

import structure
import constants

# pars = np.load('out/all_scaled/pars.npy')[
# 	:, np.load('out/all_scaled/valid.npy')
# 	]

pars = np.load('out/all_scaled/pars.npy')[
	:, np.load('out/all_scaled/valid.npy')
	]

specific_activity_data = [ # in micromol/min/mg
	('PGI', 212),
	('PGK', 98),
	('PGK', 480),
	('GPM', 124),
	('ENO', 158),
	('ENO', 260),
	('PYK', 124),
	('PPS', 8.9)
	]

molecular_weight_data = { # in kDa per molecule
	'PGI':62,
	'PGK':42,
	'GPM':27,
	'ENO':90,
	'PYK':51,
	'PPS':150
	}

# note for future reference: 1 kDa/molecule = 1 kg/mol

Nav = 6.022e23

da_per_gram = Nav
kda_per_mg = da_per_gram/1000/1000

micromol_per_mol = 1e6

s_per_min = 60

kcat_f_values = np.array([
	constants.RT * np.log(
		value * molecular_weight_data[rxn] * Nav / kda_per_mg / micromol_per_mol / s_per_min
		/ constants.K_STAR
		)
	for (rxn, value) in specific_activity_data
	])

kcat_f_matrix = np.row_stack([
	-structure.kcat_f_matrix[structure.reactions.index(rxn), :]
	for (rxn, value) in specific_activity_data
	])

predicted = kcat_f_matrix.dot(pars)

residuals = kcat_f_values - predicted.T

indexing = np.array([
	structure.reactions.index(rxn)
	for (rxn, value) in specific_activity_data
	])

sorting = np.argsort(indexing)

residuals = residuals[sorting, :]
indexing = indexing[sorting]

medians = np.median(residuals, 1)

(unique, unique_indices, inverse) = np.unique(
	indexing,
	return_index = True,
	return_inverse = True
	)

n_unique = unique.size

prediction_ranges = np.empty(
	n_unique,
	[
		('smallest', np.float64),
		('lower', np.float64),
		('upper', np.float64),
		('largest', np.float64),
		]
	)

FRACTION_OF_DATA = 0.5

DARK_GRAY = (0.0,)*3
MEDIUM_GRAY = (0.5,)*3
LIGHT_GRAY = (0.8,)*3

TENFOLD = constants.RT * np.log(10)
BILFOLD = constants.RT * np.log(1e9)

DPI = 900

DPI = 1000

TENFOLD_STYLE = dict(color = LIGHT_GRAY, lw = 0.5, zorder = -2)
ONEFOLD_STYLE = dict(color = MEDIUM_GRAY, lw = 0.5, zorder = -2)

TYPE_DIVIDER_STYLE = dict(color = LIGHT_GRAY, lw = 0.5, zorder = -1)
RANGE_STYLE = dict(c = DARK_GRAY, lw = 0.5)
IQR_STYLE = dict(c = DARK_GRAY, lw = 2)
MIDPOINT_STYLE = (
	dict(
		marker = 'h',
		c = DARK_GRAY,
		ms = 3,
		markeredgewidth = 0.5,
		markerfacecolor = 'w'
		)
	# dict(
	# 	marker = '|',
	# 	c = 'w',
	# 	ms = 2,
	# 	markeredgewidth = 0.55,
	# 	markerfacecolor = 'w'
	# 	)
	)

OBSERVATION_STYLE = (
	# dict(
	# 	marker = '^',
	# 	ms = 2.5,
	# 	markeredgewidth = 0,
	# 	)
	dict(
		marker = '^',
		ms = 2.5,
		markeredgewidth = 0.1,
		markeredgecolor = 'w'
		)
	)

OBSERVATION_OFFSET = 0.35

WELL_FIT_OBSERVATION_STYLE = OBSERVATION_STYLE.copy()
WELL_FIT_OBSERVATION_STYLE.update(
	c = (0.1, 0.3, 1,),
	zorder = 3
	)

POORLY_FIT_OBSERVATION_STYLE = OBSERVATION_STYLE.copy()
POORLY_FIT_OBSERVATION_STYLE.update(
	c = (0.9, 0.1, 0.1),
	zorder = 4
	)

for i, ind in enumerate(unique_indices):
	prediction_ranges[i] = tuple(np.percentile(
		residuals[ind, :],
		(
			0,
			50 - 100 * FRACTION_OF_DATA/2,
			50 + 100 * FRACTION_OF_DATA/2,
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

fig = plt.figure(figsize = (1, 6/(45+1)*(n_unique+1)), dpi = DPI) # DPI set here doesn't matter much

axes = fig.add_axes((0, 0, 1, 1))

axes.axvline(-TENFOLD, **TENFOLD_STYLE)
axes.axvline(+TENFOLD, **TENFOLD_STYLE)
axes.axvline(0, **dict(color = '0.5', lw = 0.5, zorder = -2))

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
fig.savefig('figure6.pdf', dpi = DPI)

with open('figure6_key.txt', 'w') as f:
	for indexing in unique:
		f.write(':'.join([
			structure.reactions[indexing]
			])+'\n')

# plt.boxplot(residuals)

# tenfold = constants.RT * np.log(1e1)

# plt.axhline(+tenfold)
# plt.axhline(-tenfold)

# bilfold = constants.RT * np.log(1e9)

# plt.axhline(+bilfold)
# plt.axhline(-bilfold)

# plt.savefig('temp.pdf')
