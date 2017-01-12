
from __future__ import division

import theano as th
import theano.tensor as tn

def build_functions():
	gibbs_energies = tn.dvector('Molar Gibbs energies')

	mu = tn.dscalar('Growth rate')

	k_star = tn.dscalar('Fundamental catalytic rate constant')
	RT = tn.dscalar('RT')

	forward_reaction_potential = tn.dmatrix('Forward reaction potential matrix')
	reverse_reaction_potential = tn.dmatrix('Reverse reaction potential matrix')

	forward_binding_potential = tn.dmatrix('Forward binding potential matrix')
	reverse_binding_potential = tn.dmatrix('Reverse binding potential matrix')

	reaction_forward_binding_association = tn.dmatrix('Reaction forward binding association matrix')
	reaction_reverse_binding_association = tn.dmatrix('Reaction reverse binding association matrix')

	stoich = tn.dmatrix('Stoichiometry matrix')

	glc_association = tn.dmatrix('Molar Gibbs energies from log-concentration association matrix')

	frp = forward_reaction_potential.dot(gibbs_energies)
	rrp = reverse_reaction_potential.dot(gibbs_energies)

	fbp = forward_binding_potential.dot(gibbs_energies)
	rbp = reverse_binding_potential.dot(gibbs_energies)

	denom = (
		1
		+ reaction_forward_binding_association.dot(tn.exp(fbp/RT))
		+ reaction_reverse_binding_association.dot(tn.exp(rbp/RT))
		)

	v = k_star * (tn.exp(frp/RT) - tn.exp(rrp/RT)) / denom

	glc = glc_association.dot(gibbs_energies)

	c = tn.exp(glc/RT)

	dc_dt = stoich.dot(v) - mu * c

	inputs = (
		gibbs_energies,

		mu,

		k_star,
		RT,

		forward_reaction_potential,
		reverse_reaction_potential,

		forward_binding_potential,
		reverse_binding_potential,

		reaction_forward_binding_association,
		reaction_reverse_binding_association,

		stoich,

		glc_association,
		)

	f_v = th.function(inputs, v, on_unused_input = 'ignore')

	f_dc_dt = th.function(inputs, dc_dt)

	# jac_dc_dt = tn.jacobian(dc_dt, c)

	# dglc_dt = RT * tn.exp(-glc/RT) * dc_dt

	dglc_dt = RT * tn.exp(-glc/RT) * stoich.dot(v) - RT * mu

	jac_dglc_dt = tn.jacobian(dglc_dt, gibbs_energies).dot(
		glc_association.T
		)

	# f_jac_dc_dt = th.function(inputs, jac_dc_dt)

	f_dglc_dt = th.function(inputs, dglc_dt)
	f_jac_dglc_dt = th.function(inputs, jac_dglc_dt)

	f_all = th.function(
		inputs,
		[
			v,
			dc_dt,
			dglc_dt,
			# jac_dglc_dt
			]
		)

	return (
		f_v,
		f_dc_dt,
		# f_jac_dc_dt,
		f_dglc_dt,
		f_jac_dglc_dt,
		f_all
		)

(
	reaction_rates,
	dc_dt,
	# jac_dc_dt,
	dglc_dt,
	jac_dglc_dt,
	compute_all
	) = build_functions()

import constants
import structure

args = (
	constants.MU,
	constants.K_STAR,
	constants.RT,
	structure.forward_reaction_potential_matrix,
	structure.reverse_reaction_potential_matrix,
	structure.forward_binding_potential_matrix,
	structure.reverse_binding_potential_matrix,
	structure.reaction_forward_binding_association_matrix,
	structure.reaction_reverse_binding_association_matrix,
	structure.stoich,
	structure.glc_association_matrix
	)
