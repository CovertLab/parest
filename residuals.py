
from __future__ import division

import os.path as pa
from itertools import izip

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

import problems
import structure
from data import kb
import fitting
import constants

DATATYPES_ORDERED = (
	'concentration',
	'standard_energy_of_formation',
	'relative_protein_count',
	'equilibrium',
	'forward_catalytic_rate',
	'reverse_catalytic_rate',
	'reactant_saturation',
	'product_saturation',
	)
REACTIONS_ORDERED = structure.reactions + sorted( # should probably hand-curate this
	entry.id for entry in kb.reaction
	if entry.id not in structure.reactions
	)
COMPOUNDS_ORDERED = structure.compounds

def ordering(ordered):
	return {item:i for (i, item) in enumerate(ordered)}

DATATYPE_ORDERING = ordering(DATATYPES_ORDERED)
REACTION_ORDERING = ordering(REACTIONS_ORDERED)
COMPOUND_ORDERING = ordering(COMPOUNDS_ORDERED)

REFERENCE_RULES = problems.DEFINITIONS['data_agnostic']

FRACTION_OF_DATA = 0.5

DARK_GRAY = (0.0,)*3
MEDIUM_GRAY = (0.5,)*3
LIGHT_GRAY = (0.8,)*3

TENFOLD = constants.RT * np.log(10)
BILFOLD = constants.RT * np.log(1e9)

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

def get_indexing(entries):
	indexing = np.empty(
		len(entries),
		[
			('datatype', 'i'),
			('reaction', 'i'),
			('compound', 'i')
			]
		)

	for (i, entry) in enumerate(entries):
		indexing[i] = (
			DATATYPE_ORDERING[getattr(entry, 'datatype', None)],
			REACTION_ORDERING.get(getattr(entry, 'reaction', None), -1),
			COMPOUND_ORDERING.get(getattr(entry, 'compound', None), -1),
			)

	return indexing

def get_absolute_fit_residuals(pars):
	(fm, fv, fe) = fitting.build_fitting_tensors(*REFERENCE_RULES)

	residuals = fm.dot(pars) - fv[:, None]

	indexing = get_indexing(fe)

	return residuals, indexing

def get_relative_fit_residuals(pars):
	all_residuals = []
	all_indexing = []

	for (fm, fv, fe) in fitting.build_relative_fitting_tensor_sets(*REFERENCE_RULES):
		raw_residuals = fm.dot(pars) - fv[:, None]

		medians = np.median(raw_residuals, 0)

		residuals = raw_residuals - medians[None, :]

		all_residuals.append(residuals)

		indexing = get_indexing(fe)

		all_indexing.append(indexing)

	return np.row_stack(all_residuals), np.concatenate(all_indexing)

def main():
	parser = ArgumentParser()
	parser.add_argument(
		'directory', type = str,
		)

	directory = parser.parse_args().directory

	valid = np.load(pa.join(directory, 'valid.npy'))

	pars = np.load(pa.join(directory, 'pars.npy'))[
		:, valid
		]

	abs_res, abs_ind = get_absolute_fit_residuals(pars)
	rel_res, rel_ind = get_relative_fit_residuals(pars)

	residuals = np.row_stack([abs_res, rel_res])
	indexing = np.concatenate([abs_ind, rel_ind])

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

	for i, ind in enumerate(unique_indices):
		prediction_ranges[i] = np.percentile(
			residuals[ind, :],
			(
				0,
				50 - 100 * FRACTION_OF_DATA/2,
				50 + 100 * FRACTION_OF_DATA/2,
				100
				)
			) - medians[ind]

	observations_by_prediction = []

	for i in xrange(n_unique):
		observation_indexes = np.where(inverse == i)[0]

		observations_by_prediction.append(
			-(medians[observation_indexes] - medians[unique_indices[i]])
			- medians[unique_indices[i]]
			)

	fig = plt.figure(figsize = (1, 6), dpi = DPI) # DPI set here doesn't matter much

	axes = fig.add_axes((0, 0, 1, 1))

	axes.axvline(0, **ONEFOLD_STYLE)
	axes.axvline(-TENFOLD, **TENFOLD_STYLE)
	axes.axvline(+TENFOLD, **TENFOLD_STYLE)

	def compare_datatype(d1, d2):
		if d1 == d2:
			return True

		elif {DATATYPES_ORDERED[d1], DATATYPES_ORDERED[d2]} == {'forward_catalytic_rate', 'reverse_catalytic_rate'}:
			return True

		elif {DATATYPES_ORDERED[d1], DATATYPES_ORDERED[d2]} == {'reactant_saturation', 'product_saturation'}:
			return True

		else:
			return False

	for i, inds in enumerate(unique):
		if i != 0:
			if not compare_datatype(last_inds['datatype'], inds['datatype']):
				axes.axhline(-i+0.5, **TYPE_DIVIDER_STYLE)

		last_inds = inds

	for i in xrange(n_unique):

		# style = TYPE_DIVIDER_STYLE.copy()
		# style['lw'] = 0.1

		# axes.plot(
		# 	(-BILFOLD, +BILFOLD),
		# 	(-i, -i),
		# 	**style
		# 	)

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

		# midpoint_shadow_style = MIDPOINT_STYLE.copy()
		# midpoint_shadow_style['c'] = DARK_GRAY
		# midpoint_shadow_style['markeredgewidth'] = midpoint_shadow_style['markeredgewidth'] * 1.5

		# axes.plot(0, -i, **midpoint_shadow_style)
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
	fig.savefig('residuals.pdf', dpi = DPI)

	print '{:0.2%} valid ({} of {})'.format(valid.mean(), valid.sum(), valid.size)
	print 'average (unscaled) fit: {:0.2f}'.format(np.mean(np.sum(np.abs(residuals), 0)))

	with open('residuals_key.txt', 'w') as f:
		for indexing in unique:
			f.write(':'.join([
				DATATYPES_ORDERED[indexing['datatype']] if indexing['datatype'] >= 0 else '',
				REACTIONS_ORDERED[indexing['reaction']] if indexing['reaction'] >= 0 else '',
				COMPOUNDS_ORDERED[indexing['compound']] if indexing['compound'] >= 0 else '',
				])+'\n')

	print len(unique), 'parameters'
	print inverse.size, 'observations'

if __name__ == '__main__':
	main()
