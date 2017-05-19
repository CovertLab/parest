
from __future__ import division

import numpy as np

from data import kb
import constants
import structure

def find_weight(rules_and_weights, entry):
	for rule, weight in rules_and_weights:
		if rule(entry):
			return weight

	else:
		raise Exception('Entry {} was not assigned a weight.'.format(entry))

def field_value_rule(**fields_and_values):
	def rule(entry):
		return all(
			hasattr(entry, field) and (getattr(entry, field) in values)
			for field, values in fields_and_values.viewitems()
			)

	return rule

def build_fitting_tensors(*rules_and_weights):
	if len(rules_and_weights) == 0:
		rules_and_weights = ((lambda entry: True, 1.0),)

	if any((weight < 0) for rule, weight in rules_and_weights):
		raise Exception('Weights must be non-negative.')

	# standard molar Gibbs energies of formation

	gs_weights = []

	gs_indices = []
	gs_values = []

	gs_entries = []

	for entry in kb.standard_energy_of_formation:
		try:
			i = structure.parameters.index(
				structure.GS.format(entry.compound)
				)

		except ValueError:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		gs_weights.append(w)

		v = entry.standard_energy_of_formation

		gs_indices.append(i)
		gs_values.append(v)

		gs_entries.append(entry)

	gs_weights = np.array(gs_weights)

	gs_indices = np.array(gs_indices)
	gs_values = np.array(gs_values)

	gs_mat = np.zeros((gs_values.size, structure.n_parameters))

	for i, j in enumerate(gs_indices):
		gs_mat[i, j] = 1

	gs_values = gs_values * gs_weights
	gs_mat = gs_mat * gs_weights[:, None]

	# molar Gibbs energies of log concentration

	glc_weights = []

	glc_indices = []
	glc_values = []

	glc_entries = []

	for entry in kb.concentration:
		try:
			i = structure.parameters.index(
				structure.GLC.format(entry.compound)
				)

		except ValueError:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		glc_weights.append(w)

		v = np.log(entry.concentration) * constants.RT

		glc_indices.append(i)
		glc_values.append(v)
		glc_entries.append(entry)

	glc_weights = np.array(glc_weights)

	glc_indices = np.array(glc_indices)
	glc_values = np.array(glc_values)

	glc_mat = np.zeros((glc_values.size, structure.n_parameters))

	for i, j in enumerate(glc_indices):
		glc_mat[i, j] = 1

	glc_values = glc_values * glc_weights
	glc_mat = glc_mat * glc_weights[:, None]

	# forward catalytic rates

	kcat_f_weights = []

	kcat_f_rows = []
	kcat_f_values = []

	kcat_f_entries = []

	for entry in kb.forward_catalytic_rate:
		if entry.reaction not in structure.reactions:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		kcat_f_weights.append(w)

		row = np.zeros(structure.n_parameters)

		gt_ind = structure.parameters.index(
			structure.GTE.format(entry.reaction)
			)

		row[gt_ind] = -1

		for reactant in structure.reactants_by_reaction[entry.reaction]:
			for s in xrange(reactant.stoichiometry):
				gs_ind = structure.parameters.index(
					structure.GS.format(
						reactant.compound,
						)
					)

				gb_ind = structure.parameters.index(
					structure.GBER.format(
						reactant.compound,
						s+1,
						reactant.reaction,
						)
					)

				row[gs_ind] += 1
				row[gb_ind] += 1

		kcat_f_rows.append(row)

		kcat_f_values.append(
			constants.RT * np.log(entry.k_cat / constants.K_STAR)
			)

		kcat_f_entries.append(entry)

	kcat_f_weights = np.array(kcat_f_weights)

	kcat_f_values = np.array(kcat_f_values)
	kcat_f_mat = np.zeros((len(kcat_f_rows), structure.n_parameters))

	for i, r in enumerate(kcat_f_rows):
		kcat_f_mat[i, :] += r

	kcat_f_values = kcat_f_values * kcat_f_weights
	kcat_f_mat = kcat_f_mat * kcat_f_weights[:, None]

	# reverse catalytic rates

	kcat_r_weights = []

	kcat_r_rows = []
	kcat_r_values = []

	kcat_r_entries = []

	for entry in kb.reverse_catalytic_rate:
		if entry.reaction not in structure.reactions:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		kcat_r_weights.append(w)

		row = np.zeros(structure.n_parameters)

		gt_ind = structure.parameters.index(
			structure.GTE.format(entry.reaction)
			)

		row[gt_ind] = -1

		for product in structure.products_by_reaction[entry.reaction]:
			for s in xrange(product.stoichiometry):
				gs_ind = structure.parameters.index(
					structure.GS.format(
						product.compound,
						)
					)

				gb_ind = structure.parameters.index(
					structure.GBEP.format(
						product.compound,
						s+1,
						product.reaction,
						)
					)

				row[gs_ind] += 1
				row[gb_ind] += 1

		kcat_r_rows.append(row)

		kcat_r_values.append(
			constants.RT * np.log(entry.k_cat / constants.K_STAR)
			)

		kcat_r_entries.append(entry)

	kcat_r_weights = np.array(kcat_r_weights)

	kcat_r_values = np.array(kcat_r_values)
	kcat_r_mat = np.zeros((len(kcat_r_rows), structure.n_parameters))

	for i, r in enumerate(kcat_r_rows):
		kcat_r_mat[i, :] += r

	kcat_r_values = kcat_r_values * kcat_r_weights
	kcat_r_mat = kcat_r_mat * kcat_r_weights[:, None]

	# forward saturation constants (reactants)

	KM_f_weights = []

	KM_f_rows = []
	KM_f_values = []

	KM_f_entries = []

	for entry in kb.reactant_saturation:
		if entry.reaction not in structure.reactions:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		KM_f_weights.append(w)

		gb_ind = structure.parameters.index(structure.GBER.format(
			entry.compound,
			entry.index,
			entry.reaction,
			))

		row = np.zeros(structure.n_parameters)

		row[gb_ind] = 1

		KM_f_rows.append(row)

		KM_f_values.append(constants.RT * np.log(entry.K_M))

		KM_f_entries.append(entry)

	KM_f_weights = np.array(KM_f_weights)

	KM_f_values = np.array(KM_f_values)

	KM_f_mat = np.zeros((len(KM_f_rows), structure.n_parameters))

	for i, r in enumerate(KM_f_rows):
		KM_f_mat[i] += r

	KM_f_values = KM_f_values * KM_f_weights
	KM_f_mat = KM_f_mat * KM_f_weights[:, None]

	# reverse saturation constants (products)

	KM_r_weights = []

	KM_r_rows = []
	KM_r_values = []

	KM_r_entries = []

	for entry in kb.product_saturation:
		if entry.reaction not in structure.reactions:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		KM_r_weights.append(w)

		gb_ind = structure.parameters.index(structure.GBEP.format(
			entry.compound,
			entry.index,
			entry.reaction
			))

		row = np.zeros(structure.n_parameters)

		row[gb_ind] = 1

		KM_r_rows.append(row)

		KM_r_values.append(constants.RT * np.log(entry.K_M))

		KM_r_entries.append(entry)

	KM_r_weights = np.array(KM_r_weights)

	KM_r_values = np.array(KM_r_values)

	KM_r_mat = np.zeros((len(KM_r_rows), structure.n_parameters))

	for i, r in enumerate(KM_r_rows):
		KM_r_mat[i] += r

	KM_r_values = KM_r_values * KM_r_weights
	KM_r_mat = KM_r_mat * KM_r_weights[:, None]

	fitting_values = np.concatenate([
		gs_values,
		glc_values,
		kcat_f_values,
		kcat_r_values,
		KM_f_values,
		KM_r_values,
		])

	fitting_mat = np.concatenate([
		gs_mat,
		glc_mat,
		kcat_f_mat,
		kcat_r_mat,
		KM_f_mat,
		KM_r_mat,
		])

	fitting_entries = sum([
		gs_entries,
		glc_entries,
		kcat_f_entries,
		kcat_r_entries,
		KM_f_entries,
		KM_r_entries,
		], [])

	return fitting_mat, fitting_values, fitting_entries

def build_relative_fitting_tensor_sets(*rules_and_weights):
	if len(rules_and_weights) == 0:
		rules_and_weights = ((lambda entry: True, 1.0),)

	if any((weight < 0) for rule, weight in rules_and_weights):
		raise Exception('Weights must be non-negative.')

	# relative enzyme concentrations

	relative_protein_count_sets = structure.gather(
		kb.relative_protein_count,
		'source'
		)

	tensor_sets = []

	for source, entries in relative_protein_count_sets.viewitems():
		gelc_weights = []
		gelc_indices = []
		gelc_values = []
		gelc_entries = []

		for entry in entries:
			if entry.reaction not in structure.reactions:
				continue

			w = find_weight(rules_and_weights, entry)

			if w == 0:
				continue

			gelc_weights.append(w)

			i = structure.parameters.index(
				structure.GELC.format(entry.reaction)
				)

			gelc_indices.append(i)

			v = constants.RT * np.log(entry.count)

			gelc_values.append(v)

			gelc_entries.append(entry)

		if len(gelc_values) == 0:
			continue

		gelc_weights = np.array(gelc_weights)

		if not np.all(gelc_weights[0] == gelc_weights):
			raise NotImplementedError(
				'Non-trivial weighted relative values not implemented - need to develop the math and methods'
				)

		gelc_indices = np.array(gelc_indices)
		gelc_values = np.array(gelc_values) * gelc_weights

		gelc_mat = np.zeros((gelc_values.size, structure.n_parameters))

		for (i, j) in enumerate(gelc_indices):
			gelc_mat[i, j] = 1

		gelc_mat = gelc_mat * gelc_weights[:, None]

		tensor_sets.append((gelc_mat, gelc_values, gelc_entries))

	return tensor_sets

def test():
	(fv, fm, fe) = build_fitting_tensors()
	tensor_sets = build_relative_fitting_tensor_sets()

if __name__ == '__main__':
	test()
