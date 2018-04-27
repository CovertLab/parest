
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
	directory = pa.join('out', 'all_scaled')

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

	import utils.residuals

	fig = utils.residuals.plot(residuals, indexing)

	fig.savefig('figure5.pdf', dpi = DPI)

	# print '{:0.2%} valid ({} of {})'.format(valid.mean(), valid.sum(), valid.size)
	# print 'average (unscaled) fit: {:0.2f}'.format(np.mean(np.sum(np.abs(residuals), 0)))

	unique = np.unique(indexing)

	with open('figure5_key.txt', 'w') as f:
		for unique_index in unique:
			f.write(':'.join([
				DATATYPES_ORDERED[unique_index['datatype']] if unique_index['datatype'] >= 0 else '',
				REACTIONS_ORDERED[unique_index['reaction']] if unique_index['reaction'] >= 0 else '',
				COMPOUNDS_ORDERED[unique_index['compound']] if unique_index['compound'] >= 0 else '',
				])+'\n')

	print len(unique), 'parameters'
	print indexing.size, 'observations'

if __name__ == '__main__':
	main()
