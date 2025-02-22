
'''
Defines the various lists, vectors, and matrices needed to connect parameter
value data to the kinetic models.

Note on data types:  While nearly all coefficients in these matrices are
integer-valued, we represent them as floating-point matrices to minimize the
amount of implicit casting when we later use these matrices in concert with
floating-point vectors.

TODO: move some of these derived/grouped matrices into another downstream file
TODO: reduce the number of public attributes and maybe write in CAPS to indicate constancy
TODO: more descriptions of what the matrices are and how to use them

'''


from __future__ import division

import numpy as np

from data import kb

DYNAMIC_COMPOUNDS = ('F6P', 'F16P', 'DHAP', 'GAP', '13DPG', '3PG', '2PG', 'PEP')
ACTIVE_REACTIONS = ('PGI', 'PFK', 'FBP', 'FBA', 'TPI', 'GAP', 'PGK', 'GPM', 'ENO', 'PYK', 'PPS') # TODO: lowercase for reaction names
N_DYNAMIC = len(DYNAMIC_COMPOUNDS)

GS = 'Gibbs standard molar energy for compound:{}'
GLC = 'Gibbs molar energy from concentration for compound:{}'
GTE = 'Gibbs molar transition energy for reaction:{}'
GELC = 'Gibbs molar energy from enzyme concentration for reaction:{}'
GBER = 'Gibbs molar binding energy for reactant compound:{}, #{} in reaction:{}'
GBEP = 'Gibbs molar binding energy for product compound:{}, #{} in reaction:{}'

# Filter for active reactions, compounds

_active_compounds = set()
for reactant in kb.reactant:
	if reactant.reaction in ACTIVE_REACTIONS:
		_active_compounds.add(reactant.compound)

for product in kb.product:
	if product.reaction in ACTIVE_REACTIONS:
		_active_compounds.add(product.compound)

n_reactions = len(ACTIVE_REACTIONS)

# Gather parameters

parameters = []

compounds = []

for compound in kb.compound:
	if compound.id in _active_compounds:
		parameters.append(GS.format(compound.id))
		parameters.append(GLC.format(compound.id))

		compounds.append(compound.id)

for reaction in kb.reaction:
	if reaction.id in ACTIVE_REACTIONS:
		parameters.append(GTE.format(reaction.id))
		parameters.append(GELC.format(reaction.id))

for reactant in kb.reactant:
	if reactant.reaction in ACTIVE_REACTIONS:
		for i in xrange(reactant.stoichiometry):
			parameters.append(GBER.format(reactant.compound, i+1, reactant.reaction))

for product in kb.product:
	if product.reaction in ACTIVE_REACTIONS:
		for i in xrange(product.stoichiometry):
			parameters.append(GBEP.format(product.compound, i+1, product.reaction))

# Build linear expressions

from collections import defaultdict
def gather(dataset, attribute):
	gathered = defaultdict(set)

	for entry in dataset:
		gathered[getattr(entry, attribute)].add(entry)

	return gathered

def all_subsets(iterable, include_empty = False):
	empty = []

	subsets = [empty]

	for item in list(iterable):
		new_subsets = []
		for subset in subsets:
			new_subsets.append(
				subset + [item]
				)

		subsets.extend(new_subsets)

	if not include_empty:
		subsets.remove(empty)

	return subsets

reactants_by_reaction = gather(kb.reactant, 'reaction')
products_by_reaction = gather(kb.product, 'reaction')

n_parameters = len(parameters)

forward_reaction_potentials = []
reverse_reaction_potentials = []
forward_binding_potentials = []
reverse_binding_potentials = []
free_energy_differences = []

solo_forward_binding_potentials = []
solo_reverse_binding_potentials = []

total_forward_binding_potentials = []
total_reverse_binding_potentials = []

reaction_stoich = []
reaction_forward_binding_associations = []
reaction_reverse_binding_associations = []

reactions = []

ind_fbp = 0
ind_rbp = 0
for ind_reaction, reaction in enumerate(kb.reaction):
	if reaction.id not in ACTIVE_REACTIONS:
		continue

	reactions.append(reaction.id)

	reactants = reactants_by_reaction[reaction.id]
	products = products_by_reaction[reaction.id]

	frp = np.zeros(n_parameters)
	rrp = np.zeros(n_parameters)
	fed = np.zeros(n_parameters)

	rs = np.zeros(N_DYNAMIC)

	i_gte = parameters.index(GTE.format(reaction.id))
	i_gelc = parameters.index(GELC.format(reaction.id))

	frp[i_gte] = +1.0
	frp[i_gelc] = -1.0

	rrp[i_gte] = +1.0
	rrp[i_gelc] = -1.0

	solo_fbp = []
	for reactant in reactants:
		s = reactant.stoichiometry
		sf = float(s)

		i_gs = parameters.index(GS.format(reactant.compound))
		i_glc = parameters.index(GLC.format(reactant.compound))

		frp[i_gs] += -sf
		frp[i_glc] += -sf

		fed[i_gs] += -sf
		fed[i_glc] += -sf

		if reactant.compound in DYNAMIC_COMPOUNDS:
			rs[DYNAMIC_COMPOUNDS.index(reactant.compound)] -= sf

		for i in xrange(s):
			fbp = np.zeros(n_parameters)

			i_gber = parameters.index(GBER.format(reactant.compound, i+1, reactant.reaction))

			fbp[i_glc] += -1.0
			fbp[i_gber] += +1.0

			solo_fbp.append(fbp)

	solo_forward_binding_potentials.extend(solo_fbp)
	total_forward_binding_potentials.append(sum(solo_fbp, np.zeros(n_parameters)))

	for fbp_subset in all_subsets(solo_fbp):
		fbp = np.sum(fbp_subset, 0)

		forward_binding_potentials.append(fbp)

		reaction_forward_binding_associations.append((ind_reaction, ind_fbp))
		ind_fbp += 1

	solo_rbp = []
	for product in products:
		s = product.stoichiometry
		sf = float(s)

		i_gs = parameters.index(GS.format(product.compound))
		i_glc = parameters.index(GLC.format(product.compound))

		rrp[i_gs] += -sf
		rrp[i_glc] += -sf

		fed[i_gs] += +sf
		fed[i_glc] += +sf

		if product.compound in DYNAMIC_COMPOUNDS:
			rs[DYNAMIC_COMPOUNDS.index(product.compound)] += sf

		for i in xrange(s):
			rbp = np.zeros(n_parameters)

			i_gbep = parameters.index(GBEP.format(product.compound, i+1, product.reaction))

			rbp[i_glc] += -1.0
			rbp[i_gbep] += +1.0

			solo_rbp.append(rbp)

	solo_reverse_binding_potentials.extend(solo_rbp)
	total_reverse_binding_potentials.append(sum(solo_rbp, np.zeros(n_parameters)))

	for rbp_subset in all_subsets(solo_rbp):
		rbp = np.sum(rbp_subset, 0)

		reverse_binding_potentials.append(rbp)

		reaction_reverse_binding_associations.append((ind_reaction, ind_rbp))
		ind_rbp += 1

	forward_reaction_potentials.append(frp)
	reverse_reaction_potentials.append(rrp)
	free_energy_differences.append(fed)

	reaction_stoich.append(rs)

forward_reaction_potential_matrix = np.array(forward_reaction_potentials)
reverse_reaction_potential_matrix = np.array(reverse_reaction_potentials)
forward_binding_potential_matrix = np.array(forward_binding_potentials)
reverse_binding_potential_matrix = np.array(reverse_binding_potentials)
free_energy_difference_matrix = np.array(free_energy_differences)

solo_forward_binding_potential_matrix = np.array(solo_forward_binding_potentials)
solo_reverse_binding_potential_matrix = np.array(solo_reverse_binding_potentials)

total_forward_binding_potential_matrix = np.array(total_forward_binding_potentials)
total_reverse_binding_potential_matrix = np.array(total_reverse_binding_potentials)

forward_saturated_reaction_potential_matrix = forward_reaction_potential_matrix - total_forward_binding_potential_matrix
reverse_saturated_reaction_potential_matrix = reverse_reaction_potential_matrix - total_reverse_binding_potential_matrix

stoich = np.array(reaction_stoich).T.copy()

reaction_forward_binding_association_matrix = np.zeros(
	(n_reactions, forward_binding_potential_matrix.shape[0])
	)
reaction_forward_binding_association_matrix[zip(*reaction_forward_binding_associations)] = 1.0

reaction_reverse_binding_association_matrix = np.zeros(
	(n_reactions, reverse_binding_potential_matrix.shape[0])
	)
reaction_reverse_binding_association_matrix[zip(*reaction_reverse_binding_associations)] = 1.0

glc_association_matrix = np.zeros((len(DYNAMIC_COMPOUNDS), n_parameters))

for i, compound in enumerate(DYNAMIC_COMPOUNDS):
	j = parameters.index(GLC.format(compound))

	glc_association_matrix[i, j] = +1.0

full_glc_association_matrix = np.zeros((len(compounds), n_parameters))

for i, compound in enumerate(compounds):
	j = parameters.index(GLC.format(compound))

	full_glc_association_matrix[i, j] = +1.0

gelc_association_matrix = np.zeros((len(reactions), n_parameters))

for i, reaction in enumerate(reactions):
	j = parameters.index(GELC.format(reaction))

	gelc_association_matrix[i, j] = +1.0

molar_masses = gather(kb.molar_mass, 'compound')

dynamic_molar_masses = []

for compound_id in DYNAMIC_COMPOUNDS:
	(entry,) = molar_masses[compound_id]

	dynamic_molar_masses.append(entry.molar_mass)

dynamic_molar_masses = np.array(dynamic_molar_masses)

activity_matrix = np.concatenate([
	forward_saturated_reaction_potential_matrix,
	reverse_saturated_reaction_potential_matrix,
	solo_forward_binding_potential_matrix,
	solo_reverse_binding_potential_matrix,
	glc_association_matrix,
	])

n_activities = activity_matrix.shape[0]

# kcat_f = exp(-kcat_f_matrix.dot(x)/RT)

kcat_f_matrix = np.zeros((n_reactions, n_parameters))

for i, reaction in enumerate(reactions):
	gt_ind = parameters.index(GTE.format(reaction))

	kcat_f_matrix[i, gt_ind] = +1.0

	for reactant in reactants_by_reaction[reaction]:
		for s in xrange(reactant.stoichiometry):
			gs_ind = parameters.index(
				GS.format(
					reactant.compound,
					)
				)

			gb_ind = parameters.index(
				GBER.format(
					reactant.compound,
					s+1,
					reactant.reaction
					)
				)

			kcat_f_matrix[i, gs_ind] -= 1.0
			kcat_f_matrix[i, gb_ind] -= 1.0

# kcat_r = exp(-kcat_r_matrix.dot(x)/RT)

kcat_r_matrix = np.zeros((n_reactions, n_parameters))

for i, reaction in enumerate(reactions):
	gt_ind = parameters.index(GTE.format(reaction))

	kcat_r_matrix[i, gt_ind] = +1.0

	for product in products_by_reaction[reaction]:
		for s in xrange(product.stoichiometry):
			gs_ind = parameters.index(
				GS.format(
					product.compound,
					)
				)

			gb_ind = parameters.index(
				GBEP.format(
					product.compound,
					s+1,
					product.reaction
					)
				)

			kcat_r_matrix[i, gs_ind] -= 1.0
			kcat_r_matrix[i, gb_ind] -= 1.0

# Keq = exp(-Keq_matrix.dot(x)/RT)

Keq_matrix = np.zeros((n_reactions, n_parameters))

for i, reaction in enumerate(reactions):
	for reactant in reactants_by_reaction[reaction]:
		j = parameters.index(
			GS.format(reactant.compound)
			)

		Keq_matrix[i, j] -= reactant.stoichiometry
	for product in products_by_reaction[reaction]:
		j = parameters.index(
			GS.format(product.compound)
			)

		Keq_matrix[i, j] += product.stoichiometry

# KM_f = exp(-KM_f_matrix.dot(x)/RT)

KM_f_matrix = np.zeros((
	solo_forward_binding_potential_matrix.shape[0],
	n_parameters
	))

KM_f_ids = []

i = 0
for reaction in reactions:
	for reactant in reactants_by_reaction[reaction]:
		for s in xrange(reactant.stoichiometry):
			gb_ind = parameters.index(
				GBER.format(
					reactant.compound,
					s+1,
					reactant.reaction
					)
				)

			KM_f_matrix[i, gb_ind] = -1.0

			KM_f_ids.append(
				'{}:{}, #{}'.format(
					reactant.reaction,
					reactant.compound,
					(s+1),
					)
				)

			i += 1

# KM_r = exp(-KM_r_matrix.dot(x)/RT)

KM_r_matrix = np.zeros((
	solo_reverse_binding_potential_matrix.shape[0],
	n_parameters
	))

KM_r_ids = []

i = 0
for reaction in reactions:
	for product in products_by_reaction[reaction]:
		for s in xrange(product.stoichiometry):
			gb_ind = parameters.index(
				GBEP.format(
					product.compound,
					s+1,
					product.reaction
					)
				)

			KM_r_matrix[i, gb_ind] = -1.0

			KM_r_ids.append(
				'{}:{}, #{}'.format(
					product.reaction,
					product.compound,
					(s+1),
					)
				)

			i += 1

gs_association_matrix = np.zeros((len(compounds), n_parameters))

for (i, compound) in enumerate(compounds):
	j = parameters.index(GS.format(compound))

	gs_association_matrix[i, j] = +1.0

standard_parameter_matrix = np.concatenate([
	full_glc_association_matrix,
	gelc_association_matrix,
	kcat_f_matrix,
	KM_f_matrix,
	KM_r_matrix,
	gs_association_matrix,
	])
