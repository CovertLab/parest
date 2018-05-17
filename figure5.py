
from __future__ import division

import os
from itertools import izip
import shutil

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

REACTIONS_ORDERED = (
	'PGI',
	'PFK',
	'PFK_no_H',
	'FBP',
	'FBA',
	'FBA_TPI_reverse',
	'TPI',
	'TPI_reverse',
	'GAP',
	'GAP_PGK',
	'PGK',
	'PGK_reverse',
	'GPM',
	'GPM_reverse',
	'ENO',
	'ENO_no_H2O',
	'PYK',
	'PYK_no_H_reverse',
	'PPS'
	)

REVERSED_REACTIONS = (
	'FBA_TPI_reverse',
	'TPI_reverse',
	'PGK_reverse',
	'GPM_reverse',
	'PYK_no_H_reverse',
	)

COMPOUNDS_ORDERED = structure.compounds

def ordering(ordered):
	return {item:i for (i, item) in enumerate(ordered)}

DATATYPE_ORDERING = ordering(DATATYPES_ORDERED)

# not concerned about the distinction
DATATYPE_ORDERING['reverse_catalytic_rate'] = DATATYPE_ORDERING['forward_catalytic_rate']
DATATYPE_ORDERING['product_saturation'] = DATATYPE_ORDERING['reactant_saturation']

REACTION_ORDERING = ordering(REACTIONS_ORDERED)
COMPOUND_ORDERING = ordering(COMPOUNDS_ORDERED)

REFERENCE_RULES = problems.DEFINITIONS['data_agnostic']

DPI = 1000

def get_indexing(entries):
	indexing = np.empty(
		len(entries),
		[
			('datatype', 'i'),
			('reaction', 'i'),
			('compound', 'i')
			]
		)

	do_reverse = np.zeros(len(entries), np.bool)

	for (i, entry) in enumerate(entries):
		indexing[i] = (
			DATATYPE_ORDERING[getattr(entry, 'datatype', None)],
			REACTION_ORDERING.get(getattr(entry, 'reaction', None), -1),
			COMPOUND_ORDERING.get(getattr(entry, 'compound', None), -1),
			)

		do_reverse[i] = (getattr(entry, 'reaction', None) in REVERSED_REACTIONS)

	return indexing, do_reverse

def get_absolute_fit_residuals(pars):
	(fm, fv, fe) = fitting.build_fitting_tensors(*REFERENCE_RULES)

	residuals = fm.dot(pars) - fv[:, None]

	indexing, do_reverse = get_indexing(fe)

	residuals[do_reverse] = -residuals[do_reverse]

	return residuals, indexing

def get_relative_fit_residuals(pars):
	all_residuals = []
	all_indexing = []

	for (fm, fv, fe) in fitting.build_relative_fitting_tensor_sets(*REFERENCE_RULES):
		raw_residuals = fm.dot(pars) - fv[:, None]

		medians = np.median(raw_residuals, 0)

		residuals = raw_residuals - medians[None, :]

		all_residuals.append(residuals)

		indexing, do_reverse = get_indexing(fe)

		assert not np.any(do_reverse)

		all_indexing.append(indexing)

	return np.row_stack(all_residuals), np.concatenate(all_indexing)

def make_clean_directory(directory):
	if os.path.exists(directory):
		shutil.rmtree(directory)

	os.makedirs(directory)

def main(input_directory, output_directory):
	valid = np.load(os.path.join(input_directory, 'valid.npy'))

	pars = np.load(os.path.join(input_directory, 'pars.npy'))[
		:, valid
		]

	abs_res, abs_ind = get_absolute_fit_residuals(pars)
	rel_res, rel_ind = get_relative_fit_residuals(pars)

	residuals = np.row_stack([abs_res, rel_res])
	indexing = np.concatenate([abs_ind, rel_ind])

	sorting = np.argsort(indexing)

	residuals = residuals[sorting, :]
	indexing = indexing[sorting]

	datatypes = indexing['datatype']

	import utils.residuals

	make_clean_directory(output_directory)

	for unique_datatype_index in np.unique(datatypes):
		plotted = (datatypes == unique_datatype_index)

		name = DATATYPES_ORDERED[unique_datatype_index]

		fig = utils.residuals.plot(residuals[plotted], indexing[plotted])

		fig.savefig(os.path.join(output_directory, '{}.pdf'.format(name)), dpi = DPI)

		# print '{:0.2%} valid ({} of {})'.format(valid.mean(), valid.sum(), valid.size)

		# WARNING: This is the error in the 'data agnostic' problem, not the 'all scaled' problem
		# print 'average (unscaled) fit: {:0.2f}'.format(np.mean(np.sum(np.abs(residuals), 0)))

		plt.close(fig)

	unique = np.unique(indexing)

	with open(os.path.join(output_directory, 'key.txt'), 'w') as f:
		for unique_index in unique:
			f.write(':'.join([
				DATATYPES_ORDERED[unique_index['datatype']] if unique_index['datatype'] >= 0 else '',
				REACTIONS_ORDERED[unique_index['reaction']] if unique_index['reaction'] >= 0 else '',
				COMPOUNDS_ORDERED[unique_index['compound']] if unique_index['compound'] >= 0 else '',
				])+'\n')

	print len(unique), 'parameters'
	print indexing.size, 'observations'

if __name__ == '__main__':
	main(
		os.path.join('out', 'all_scaled'),
		'figure5'
		)
