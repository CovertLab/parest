
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
			(getattr(entry, field) in values)
			for field, values in fields_and_values.viewitems()
			)

	return rule

def build_fitting_tensors(*rules_and_weights):
	if len(rules_and_weights) == 0:
		rules_and_weights = ((lambda entry: True, 1.0),)

	# standard molar Gibbs energies of formation

	gs_weights = []

	gs_indices = []
	gs_values = []

	gs_ids = []

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

		gs_ids.append('{}({})'.format(entry.datatype, entry.id))

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

	glc_ids = []

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
		glc_ids.append('{}({})'.format(entry.datatype, entry.id))

	glc_weights = np.array(glc_weights)

	glc_indices = np.array(glc_indices)
	glc_values = np.array(glc_values)

	glc_mat = np.zeros((glc_values.size, structure.n_parameters))

	for i, j in enumerate(glc_indices):
		glc_mat[i, j] = 1

	glc_values = glc_values * glc_weights
	glc_mat = glc_mat * glc_weights[:, None]

	# forward catalytic rates

	kf_weights = []

	kf_rows = []
	kf_values = []

	kf_ids = []

	for entry in kb.forward_catalytic_rate:
		if entry.reaction not in structure.reactions:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		kf_weights.append(w)

		row = np.zeros(structure.n_parameters)

		gt_ind = structure.parameters.index(
			structure.GTE.format(entry.reaction)
			)

		row[gt_ind] = -1

		for reactant in structure.reactants_by_reaction[entry.reaction]:
			for s in xrange(reactant.stoichiometry):
				gb_ind = structure.parameters.index(
					structure.GBER.format(
						reactant.compound,
						s+1,
						reactant.reaction,
						)
					)

				row[gb_ind] += 1

		kf_rows.append(row)

		kf_values.append(
			constants.RT * np.log(entry.k_cat / constants.K_STAR)
			)

		kf_ids.append('{}({})'.format(entry.datatype, entry.id))

	kf_weights = np.array(kf_weights)

	kf_values = np.array(kf_values)
	kf_mat = np.zeros((len(kf_rows), structure.n_parameters))

	for i, r in enumerate(kf_rows):
		kf_mat[i, :] += r

	kf_values = kf_values * kf_weights
	kf_mat = kf_mat * kf_weights[:, None]

	# reverse catalytic rates

	kr_weights = []

	kr_rows = []
	kr_values = []

	kr_ids = []

	for entry in kb.reverse_catalytic_rate:
		if entry.reaction not in structure.reactions:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		kr_weights.append(w)

		row = np.zeros(structure.n_parameters)

		gt_ind = structure.parameters.index(
			structure.GTE.format(entry.reaction)
			)

		row[gt_ind] = -1

		for product in structure.products_by_reaction[entry.reaction]:
			for s in xrange(product.stoichiometry):
				gb_ind = structure.parameters.index(
					structure.GBEP.format(
						product.compound,
						s+1,
						product.reaction,
						)
					)

				row[gb_ind] += 1

		kr_rows.append(row)

		kr_values.append(
			constants.RT * np.log(entry.k_cat / constants.K_STAR)
			)

		kr_ids.append('{}({})'.format(entry.datatype, entry.id))

	kr_weights = np.array(kr_weights)

	kr_values = np.array(kr_values)
	kr_mat = np.zeros((len(kr_rows), structure.n_parameters))

	for i, r in enumerate(kr_rows):
		kr_mat[i, :] += r

	kr_values = kr_values * kr_weights
	kr_mat = kr_mat * kr_weights[:, None]

	# saturation constants

	# things are a little hokey here because saturations in given rate laws don't
	# specify the identity of a given substrate in a saturation term
	# improving the data format would make this logic cleaner

	KM_weights = []

	KM_rows = []
	KM_values = []

	KM_ids = []

	for entry in kb.substrate_saturation:
		if entry.reaction not in structure.reactions:
			continue

		w = find_weight(rules_and_weights, entry)

		if w == 0:
			continue

		KM_weights.append(w)

		try:
			gb_ind = structure.parameters.index(structure.GBER.format(
				entry.compound,
				1,
				entry.reaction,
				))

		except ValueError:
			try:
				gb_ind = structure.parameters.index(structure.GBEP.format(
					entry.compound,
					1,
					entry.reaction
					))

			except ValueError:
				raise Exception('could not find a corresponding parameter for {}'.format(entry))

		gs_ind = structure.parameters.index(structure.GS.format(entry.compound))

		row = np.zeros(structure.n_parameters)

		row[gb_ind] = -1
		row[gs_ind] = +1

		KM_rows.append(row)

		KM_values.append(constants.RT * np.log(entry.K_M))

		KM_ids.append('{}({})'.format(entry.datatype, entry.id))

	KM_weights = np.array(KM_weights)

	KM_values = np.array(KM_values)

	KM_mat = np.zeros((len(KM_rows), structure.n_parameters))

	for i, r in enumerate(KM_rows):
		KM_mat[i] += r

	KM_values = KM_values * KM_weights
	KM_mat = KM_mat * KM_weights[:, None]

	# TODO: relative enzyme concentrations

	# TODO: relative fluxes

	fitting_values = np.concatenate([
		gs_values,
		glc_values,
		kf_values,
		kr_values,
		KM_values
		])

	fitting_mat = np.concatenate([
		gs_mat,
		glc_mat,
		kf_mat,
		kr_mat,
		KM_mat
		])

	fitting_ids = np.concatenate([
		gs_ids,
		glc_ids,
		kf_ids,
		kr_ids,
		KM_ids
		])

	return fitting_mat, fitting_values, fitting_ids
