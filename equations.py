
'''

Defines the equations for the modeled dynamical system.  Most of the arguments
to these functions are matrices, generated by structure.py.

Call pattern:

(v, dc_dt, dglc_dt) = equations.compute_all(parameter_values, *equations.args)

'''


from __future__ import division

import numpy as np

def reaction_rates(
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
		):
	'''
	Computes net reaction rates.
	'''

	frp = forward_reaction_potential.dot(gibbs_energies)
	rrp = reverse_reaction_potential.dot(gibbs_energies)

	fbp = forward_binding_potential.dot(gibbs_energies)
	rbp = reverse_binding_potential.dot(gibbs_energies)

	denom = (
		1
		+ reaction_forward_binding_association.dot(np.exp(-fbp/RT))
		+ reaction_reverse_binding_association.dot(np.exp(-rbp/RT))
		)

	return k_star * (np.exp(-frp/RT) - np.exp(-rrp/RT)) / denom

def dc_dt(
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
		):
	'''
	Computes dc/dt, where 'c' is the metabolite concentrations.
	'''

	frp = forward_reaction_potential.dot(gibbs_energies)
	rrp = reverse_reaction_potential.dot(gibbs_energies)

	fbp = forward_binding_potential.dot(gibbs_energies)
	rbp = reverse_binding_potential.dot(gibbs_energies)

	denom = (
		1
		+ reaction_forward_binding_association.dot(np.exp(-fbp/RT))
		+ reaction_reverse_binding_association.dot(np.exp(-rbp/RT))
		)

	v = k_star * (np.exp(-frp/RT) - np.exp(-rrp/RT)) / denom

	glc = glc_association.dot(gibbs_energies)

	c = np.exp(glc/RT)

	return stoich.dot(v) - mu * c

def dglc_dt(
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
		):
	'''
	Computes dglc/dt, where 'glc' is the Gibbs energy from the logarithm on the
	metabolite concentrations.
	'''

	frp = forward_reaction_potential.dot(gibbs_energies)
	rrp = reverse_reaction_potential.dot(gibbs_energies)

	fbp = forward_binding_potential.dot(gibbs_energies)
	rbp = reverse_binding_potential.dot(gibbs_energies)

	denom = (
		1
		+ reaction_forward_binding_association.dot(np.exp(-fbp/RT))
		+ reaction_reverse_binding_association.dot(np.exp(-rbp/RT))
		)

	v = k_star * (np.exp(-frp/RT) - np.exp(-rrp/RT)) / denom

	glc = glc_association.dot(gibbs_energies)

	return RT * np.exp(-glc/RT) * stoich.dot(v) - RT * mu

def compute_all(
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
		):
	'''
	Computes all the terms needed for the optimization.  This is done in one
	function rather than calling the individual functions because it can reuse
	several intermediate calculations.
	'''

	frp = forward_reaction_potential.dot(gibbs_energies)
	rrp = reverse_reaction_potential.dot(gibbs_energies)

	fbp = forward_binding_potential.dot(gibbs_energies)
	rbp = reverse_binding_potential.dot(gibbs_energies)

	denom = (
		1
		+ reaction_forward_binding_association.dot(np.exp(-fbp/RT))
		+ reaction_reverse_binding_association.dot(np.exp(-rbp/RT))
		)

	v = k_star * (np.exp(-frp/RT) - np.exp(-rrp/RT)) / denom

	glc = glc_association.dot(gibbs_energies)

	c = np.exp(glc/RT)

	dc = stoich.dot(v) - mu * c

	dglc = RT * np.exp(-glc/RT) * stoich.dot(v) - RT * mu

	return (v, dc, dglc)

def forward_reaction_rates(
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
		):
	'''
	Computes the forward rates of reaction.
	'''

	frp = forward_reaction_potential.dot(gibbs_energies)
	# rrp = reverse_reaction_potential.dot(gibbs_energies)

	fbp = forward_binding_potential.dot(gibbs_energies)
	rbp = reverse_binding_potential.dot(gibbs_energies)

	denom = (
		1
		+ reaction_forward_binding_association.dot(np.exp(-fbp/RT))
		+ reaction_reverse_binding_association.dot(np.exp(-rbp/RT))
		)

	return k_star * np.exp(-frp/RT) / denom

def reverse_reaction_rates(
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
		):

	'''
	Computes the reverse rates of reaction.
	'''

	# frp = forward_reaction_potential.dot(gibbs_energies)
	rrp = reverse_reaction_potential.dot(gibbs_energies)

	fbp = forward_binding_potential.dot(gibbs_energies)
	rbp = reverse_binding_potential.dot(gibbs_energies)

	denom = (
		1
		+ reaction_forward_binding_association.dot(np.exp(-fbp/RT))
		+ reaction_reverse_binding_association.dot(np.exp(-rbp/RT))
		)

	return k_star * np.exp(-rrp/RT) / denom

def build_jacobian():
	'''
	This function uses Theano, a symbolic math library, to analytically
	evaluate the Jacobian of the dynamical system.  In the spirit of minimizing
	dependencies, and due to some performance issues, I opted to not use Theano
	for this work.  However I retained this function for legacy purposes.
	'''

	import theano as th
	import theano.tensor as tn

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
		+ reaction_forward_binding_association.dot(tn.exp(-fbp/RT))
		+ reaction_reverse_binding_association.dot(tn.exp(-rbp/RT))
		)

	v = k_star * (tn.exp(-frp/RT) - tn.exp(-rrp/RT)) / denom

	glc = glc_association.dot(gibbs_energies)

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

	dglc_dt = RT * tn.exp(-glc/RT) * stoich.dot(v) - RT * mu

	jac_dglc_dt = tn.jacobian(dglc_dt, gibbs_energies).dot(
		glc_association.T
		)

	f_jac_dglc_dt = th.function(inputs, jac_dglc_dt)

	return f_jac_dglc_dt

import constants
import structure

'''
args is the set of arguments passed to the equations above that aren't
parameter values.
'''

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
