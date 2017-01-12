
from __future__ import division

import numpy as np

from data import kb

DYNAMIC_COMPOUNDS = ('F6P', 'F16P', 'DHAP', 'GAP', '13DPG', '3PG', '2PG', 'PEP')
N_DYNAMIC = len(DYNAMIC_COMPOUNDS)

GS = 'Gibbs standard molar energy for compound:{}'
GLC = 'Gibbs molar energy from concentration for compound:{}'
GTE = 'Gibbs molar transition energy for reaction:{}'
GELC = 'Gibbs molar energy from enzyme concentration for reaction:{}'
GBER = 'Gibbs molar binding energy for reactant compound:{}, #{} in reaction:{}'
GBEP = 'Gibbs molar binding energy for product compound:{}, #{} in reaction:{}'

# Filter for active reactions, compounds

active_reactions = set()
for reactant in kb.reactant:
	if reactant.compound in DYNAMIC_COMPOUNDS:
		active_reactions.add(reactant.reaction)

for product in kb.product:
	if product.compound in DYNAMIC_COMPOUNDS:
		active_reactions.add(product.reaction)

active_compounds = set()
for reactant in kb.reactant:
	if reactant.reaction in active_reactions:
		active_compounds.add(reactant.compound)

for product in kb.product:
	if product.reaction in active_reactions:
		active_compounds.add(product.compound)

n_reactions = len(active_reactions)

# Gather parameters

parameters = []

for compound in kb.compound:
	if compound.id in active_compounds:
		parameters.append(GS.format(compound.id))
		parameters.append(GLC.format(compound.id))

for reaction in kb.reaction:
	if reaction.id in active_reactions:
		parameters.append(GTE.format(reaction.id))
		parameters.append(GELC.format(reaction.id))

for reactant in kb.reactant:
	if reactant.reaction in active_reactions:
		for i in xrange(reactant.stoichiometry):
			parameters.append(GBER.format(reactant.compound, i+1, reactant.reaction))

for product in kb.product:
	if product.reaction in active_reactions:
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
	if reaction.id not in active_reactions:
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

	frp[i_gte] = -1
	frp[i_gelc] = +1

	rrp[i_gte] = -1
	rrp[i_gelc] = +1

	solo_fbp = []
	for reactant in reactants:
		s = reactant.stoichiometry

		i_gs = parameters.index(GS.format(reactant.compound))
		i_glc = parameters.index(GLC.format(reactant.compound))

		frp[i_gs] += +s
		frp[i_glc] += +s

		fed[i_gs] += -s
		fed[i_glc] += -s

		if reactant.compound in DYNAMIC_COMPOUNDS:
			rs[DYNAMIC_COMPOUNDS.index(reactant.compound)] -= s

		for i in xrange(s):
			fbp = np.zeros(n_parameters)

			i_gber = parameters.index(GBER.format(reactant.compound, i+1, reactant.reaction))

			fbp[i_gs] += +1
			fbp[i_glc] += +1
			fbp[i_gber] += -1

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

		i_gs = parameters.index(GS.format(product.compound))
		i_glc = parameters.index(GLC.format(product.compound))

		rrp[i_gs] += +s
		rrp[i_glc] += +s

		fed[i_gs] += +s
		fed[i_glc] += +s

		if product.compound in DYNAMIC_COMPOUNDS:
			rs[DYNAMIC_COMPOUNDS.index(product.compound)] += s

		for i in xrange(s):
			rbp = np.zeros(n_parameters)

			i_gbep = parameters.index(GBEP.format(product.compound, i+1, product.reaction))

			rbp[i_gs] += +1
			rbp[i_glc] += +1
			rbp[i_gbep] += -1

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

stoich = np.array(reaction_stoich).T

reaction_forward_binding_association_matrix = np.zeros(
	(n_reactions, forward_binding_potential_matrix.shape[0])
	)
reaction_forward_binding_association_matrix[zip(*reaction_forward_binding_associations)] = 1

reaction_reverse_binding_association_matrix = np.zeros(
	(n_reactions, reverse_binding_potential_matrix.shape[0])
	)
reaction_reverse_binding_association_matrix[zip(*reaction_reverse_binding_associations)] = 1

glc_association_matrix = np.zeros((len(DYNAMIC_COMPOUNDS), n_parameters))

for i, compound in enumerate(DYNAMIC_COMPOUNDS):
	j = parameters.index(GLC.format(compound))

	glc_association_matrix[i, j] = +1

full_glc_association_matrix = np.zeros((len(active_compounds), n_parameters))

for i, compound in enumerate(active_compounds):
	j = parameters.index(GLC.format(compound))

	full_glc_association_matrix[i, j] = +1

gelc_association_matrix = np.zeros((len(active_reactions), n_parameters))

for i, reaction in enumerate(active_reactions):
	j = parameters.index(GELC.format(reaction))

	gelc_association_matrix[i, j] = +1

# potential_matrix = np.vstack([
# 	# forward_reaction_potential_matrix,
# 	# reverse_reaction_potential_matrix,

# 	forward_saturated_reaction_potential_matrix,
# 	reverse_saturated_reaction_potential_matrix,

# 	# forward_binding_potential_matrix,
# 	# reverse_binding_potential_matrix,
# 	# free_energy_difference_matrix,

# 	solo_forward_binding_potential_matrix,
# 	solo_reverse_binding_potential_matrix,

# 	full_glc_association_matrix,
# 	gelc_association_matrix,

# 	])

# def rho(x):
# 	fbp = forward_binding_potential_matrix.dot(x)
# 	rbp = reverse_binding_potential_matrix.dot(x)

# 	denom = (
# 		1
# 		+ reaction_forward_binding_association_matrix.dot(np.exp(fbp / RT))
# 		+ reaction_reverse_binding_association_matrix.dot(np.exp(rbp / RT))
# 		)

# 	return 1/denom

# def degree_of_saturation(x):
# 	return 1-rho(x)

# def reaction_rates(x):
# 	frp = forward_reaction_potential_matrix.dot(x)
# 	rrp = reverse_reaction_potential_matrix.dot(x)

# 	fbp = forward_binding_potential_matrix.dot(x)
# 	rbp = reverse_binding_potential_matrix.dot(x)

# 	denom = (
# 		1
# 		+ reaction_forward_binding_association_matrix.dot(np.exp(fbp / RT))
# 		+ reaction_reverse_binding_association_matrix.dot(np.exp(rbp / RT))
# 		)

# 	v = K_STAR * (np.exp(frp / RT) - np.exp(rrp / RT)) / denom

# 	return v

# def forward_reaction_rates(x):
# 	frp = forward_reaction_potential_matrix.dot(x)

# 	fbp = forward_binding_potential_matrix.dot(x)
# 	rbp = reverse_binding_potential_matrix.dot(x)

# 	denom = (
# 		1
# 		+ reaction_forward_binding_association_matrix.dot(np.exp(fbp / RT))
# 		+ reaction_reverse_binding_association_matrix.dot(np.exp(rbp / RT))
# 		)

# 	v_f = K_STAR * np.exp(frp / RT) / denom

# 	return v_f

# def reverse_reaction_rates(x):
# 	rrp = reverse_reaction_potential_matrix.dot(x)

# 	fbp = forward_binding_potential_matrix.dot(x)
# 	rbp = reverse_binding_potential_matrix.dot(x)

# 	denom = (
# 		1
# 		+ reaction_forward_binding_association_matrix.dot(np.exp(fbp / RT))
# 		+ reaction_reverse_binding_association_matrix.dot(np.exp(rbp / RT))
# 		)

# 	v_r = K_STAR * np.exp(rrp / RT) / denom

# 	return v_r

# def dc_dt(x):
# 	return stoich.dot(reaction_rates(x))

# S_product = stoich.copy()
# S_product[stoich < 0] = 0
# S_reactant = -stoich
# S_reactant[stoich > 0] = 0

# def accumulation_rate(x):
# 	frp = forward_reaction_potential_matrix.dot(x)
# 	rrp = reverse_reaction_potential_matrix.dot(x)

# 	fbp = forward_binding_potential_matrix.dot(x)
# 	rbp = reverse_binding_potential_matrix.dot(x)

# 	denom = (
# 		1
# 		+ reaction_forward_binding_association_matrix.dot(np.exp(fbp / RT))
# 		+ reaction_reverse_binding_association_matrix.dot(np.exp(rbp / RT))
# 		)

# 	v_f = K_STAR * np.exp(frp / RT) / denom
# 	v_r = K_STAR * np.exp(rrp / RT) / denom

# 	acc = S_product.dot(v_f) + S_reactant.dot(v_r)

# 	return acc

# def consumption_rate(x):
# 	frp = forward_reaction_potential_matrix.dot(x)
# 	rrp = reverse_reaction_potential_matrix.dot(x)

# 	fbp = forward_binding_potential_matrix.dot(x)
# 	rbp = reverse_binding_potential_matrix.dot(x)

# 	denom = (
# 		1
# 		+ reaction_forward_binding_association_matrix.dot(np.exp(fbp / RT))
# 		+ reaction_reverse_binding_association_matrix.dot(np.exp(rbp / RT))
# 		)

# 	v_f = K_STAR * np.exp(frp / RT) / denom
# 	v_r = K_STAR * np.exp(rrp / RT) / denom

# 	con = S_reactant.dot(v_f) + S_product.dot(v_r)

# 	return con

# def log_relative_excess(x):
# 	frp = forward_reaction_potential_matrix.dot(x)
# 	rrp = reverse_reaction_potential_matrix.dot(x)

# 	fbp = forward_binding_potential_matrix.dot(x)
# 	rbp = reverse_binding_potential_matrix.dot(x)

# 	denom = (
# 		1
# 		+ reaction_forward_binding_association_matrix.dot(np.exp(fbp / RT))
# 		+ reaction_reverse_binding_association_matrix.dot(np.exp(rbp / RT))
# 		)

# 	v_f = K_STAR * np.exp(frp / RT) / denom
# 	v_r = K_STAR * np.exp(rrp / RT) / denom

# 	acc = S_product.dot(v_f) + S_reactant.dot(v_r)
# 	con = S_reactant.dot(v_f) + S_product.dot(v_r)

# 	return np.log(acc) - np.log(con)

# conc_ind = [parameters.index(GLC.format(compound)) for compound in DYNAMIC_COMPOUNDS]
# is_dynamic = np.zeros(n_parameters, np.bool)
# is_dynamic[conc_ind] = True
# is_static = ~is_dynamic

# def ode_init(values):
# 	c0 = np.exp(values[is_dynamic] / RT)

# 	parameters = values[is_static]

# 	args = (parameters,)

# 	return c0, args

# def ode(c, t, *args):
# 	(parameters,) = args
# 	values = np.zeros(n_parameters)

# 	values[is_dynamic] = RT * np.log(c)
# 	values[is_static] = parameters

# 	return dc_dt(values)
