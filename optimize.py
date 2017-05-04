
from __future__ import division

from itertools import izip

import numpy as np

import liveout as lo

import structure
import fitting
import equations
import utils.linalg as la

from initialization import build_initial_parameter_values

MAX_ITERATIONS = int(1e6)
CONVERGENCE_RATE = 1e-4
CONVERGENCE_TIME = int(1e4)

PERTURB_INIT = 1e1
PERTURB_FINAL = 1e-6

OBJ_MASSEQ_WEIGHT = 1e0
OBJ_ENERGYEQ_WEIGHT = 1e0
OBJ_FLUX_WEIGHT = 1e0
OBJ_FIT_WEIGHT = 1e0

INIT_CONSTRAINT_PENALTY_WEIGHT = 1e-10
FINAL_CONSTRAINT_PENALTY_WEIGHT = 1e10
CONSTRAINT_PENALTY_GROWTH_RATE = 1e1
CONSTRAINT_PENALTY_GROWTH_ITERATIONS = int(np.ceil(
	np.log(FINAL_CONSTRAINT_PENALTY_WEIGHT / INIT_CONSTRAINT_PENALTY_WEIGHT)
	/ np.log(CONSTRAINT_PENALTY_GROWTH_RATE)
	))

TARGET_PYRUVATE_PRODUCTION = 1e-3

def fast_shortarray_median1d(x):
	# faster 1d median than np.median for short arrays (100 or less elements, up to 10x speed-up)
	# I don't know why the built-in median is so slow; possibly due to Python overhead
	size = x.size
	(halfsize, remainder) = divmod(size, 2)
	x_sorted = np.sort(x)

	if remainder == 0:
		return (x_sorted[halfsize-1] + x_sorted[halfsize])/2

	else:
		return x_sorted[halfsize]

median1d = (
	# np.median
	fast_shortarray_median1d
	)

def compute_relative_fit(x, tensor_sets):
	cost = 0

	for (fm, fv, fe) in tensor_sets:
		predicted = fm.dot(x)
		d = predicted - fv

		m = median1d(d)

		cost += np.sum(np.abs(d - m))

	return cost

class ObjectiveValues(object):
	def __init__(self, pars, fitting_tensors, relative_fitting_tensor_sets = ()):
		(v, dc_dt, dglc_dt) = equations.compute_all(pars, *equations.args)

		self.mass_eq = np.sum(np.square(structure.dynamic_molar_masses * dc_dt))
		self.energy_eq = np.sum(np.square(dglc_dt))

		net_pyruvate_production = v[-2] - v[-1]

		self.flux = (net_pyruvate_production / TARGET_PYRUVATE_PRODUCTION - 1)**2

		(fitting_matrix, fitting_values) = fitting_tensors[:2]

		self.fit = (
			np.sum(np.abs(fitting_matrix.dot(pars) - fitting_values))
			+ compute_relative_fit(pars, relative_fitting_tensor_sets)
			)

	def total(self, weight_mass_eq, weight_energy_eq, weight_flux, weight_fit):
		return (
			weight_mass_eq * self.mass_eq
			+ weight_energy_eq * self.energy_eq
			+ weight_flux * self.flux
			+ weight_fit * self.fit
			)

def nonredundant_vectors(vectors, tolerance = 1e-15):
	# Discards vectors that are equivalent to subsequent vectors under scaling.

	retained = []

	normed = [v / np.sqrt(v.dot(v)) for v in vectors]

	for i, vi in enumerate(normed):
		for vj in normed[i+1:]:
			if 1 - np.abs(vi.dot(vj)) < 1e-15:
				break

		else:
			retained.append(vectors[i])

	return retained

def build_perturbation_vectors(naive = False):
	if naive:
		perturbation_vectors = [
			vector
			for vector in np.linalg.pinv(structure.standard_parameter_matrix).T
			]

		# The standard parameter matrix corresponds to the 'usual'
		# parameterization of a kinetic model, in terms of kcat's, metabolite
		# and enzyme concentrations, saturation constants (KM's), and Gibbs
		# standard energies of formation.

		# These perturbations are 'simple' in the sense that a perturbation to
		# a given 'standard' parameter does not influence the value of any
		# other 'standard' parameter.

	else:
		perturbation_vectors = [
			vector
			for vector in la.bilevel_elementwise_pseudoinverse(
				np.concatenate([
					structure.activity_matrix,
					np.identity(structure.n_parameters),
					]),
				structure.activity_matrix
				).T
			]

		perturbation_vectors = nonredundant_vectors(perturbation_vectors)

		# Many of the basic parameter perturbation vectors end up being
		# identical to the 'activity matrix' perturbation vectors, so I detect
		# and strip those from the set.

	return perturbation_vectors

def build_bounds(naive = False):
	if naive:
		import constants

		lower_conc = 1e-10 # avg 1/10 molecules per e. coli cell
		upper_conc = 1e2 # water conc is around 40 M

		lower_enz_conc = 1e-10 # avg 1/10 molecules per cell
		upper_enz_conc = 1e-2 # total protein conc in e. coli is about 1-10 mM

		lower_kcat = 1e-5 # some mutants/unusual substrates can be very low
		upper_kcat = 1e6 # catalase is around 1e5 /s

		lower_KM = lower_conc
		upper_KM = upper_conc

		lower_glc = constants.RT * np.log(lower_conc)
		upper_glc = constants.RT * np.log(upper_conc)

		lower_gelc = constants.RT * np.log(lower_enz_conc)
		upper_gelc = constants.RT * np.log(upper_enz_conc)

		lower_log_kcat = -constants.RT * np.log(upper_kcat)
		upper_log_kcat = -constants.RT * np.log(lower_kcat)

		lower_log_KM = -constants.RT * np.log(upper_KM)
		upper_log_KM = -constants.RT * np.log(lower_KM)

		bounds_matrix = np.concatenate([
			structure.full_glc_association_matrix,
			structure.gelc_association_matrix,
			structure.kcat_f_matrix,
			structure.KM_f_matrix,
			structure.KM_r_matrix,
			])

		lowerbounds = np.concatenate([
			[lower_glc] * structure.full_glc_association_matrix.shape[0],
			[lower_gelc] * structure.gelc_association_matrix.shape[0],
			[lower_log_kcat] * structure.kcat_f_matrix.shape[0],
			[lower_log_KM] * structure.KM_f_matrix.shape[0],
			[lower_log_KM] * structure.KM_r_matrix.shape[0],
			])

		upperbounds = np.concatenate([
			[upper_glc] * structure.full_glc_association_matrix.shape[0],
			[upper_gelc] * structure.gelc_association_matrix.shape[0],
			[upper_log_kcat] * structure.kcat_f_matrix.shape[0],
			[upper_log_KM] * structure.KM_f_matrix.shape[0],
			[upper_log_KM] * structure.KM_r_matrix.shape[0],
			])

		# I'm excluding reverse kcat's because it adds some non-trivial (that
		# is, not 'naive') bounding logic, which in turn allows for non-naive
		# perturbations on the system.

		# Including reverse kcat bounds does improve the frequency of
		# convergence to some extent, but I want a clean separation between
		# the 'naive' system and my augmented approach.

		# I can't include Keq bounds because those are interdependent (and
		# therefore not 'naive').  Removing the relationship between reverse
		# and forward kcat's would fundamentally change the model in such a
		# fashion that the energy balance (non-decreasing entropy) is not
		# enforced.

		# Unlike all other 'standard' parameters, I do not bound Gibbs standard
		# energies of formation.  There are no obvious or useful bounds on
		# those values.

	else:
		import bounds

		bounds_matrix = bounds.BOUNDS_MATRIX
		lowerbounds = bounds.LOWERBOUNDS
		upperbounds = bounds.UPPERBOUNDS

	return bounds_matrix, lowerbounds, upperbounds

RESIDUAL_CUTOFF = 1e-5

def empty_callback(epoch, iteration, constraint_penalty_weight, obj_components):
	pass

def estimate_parameters(
		fitting_rules_and_weights = tuple(),
		random_state = None,
		naive = False,
		callback = empty_callback
		):

	if random_state is None:
		random_state = np.random.RandomState()

	fitting_tensors = (
		fitting_matrix,
		fitting_values,
		fitting_entries
		) = fitting.build_fitting_tensors(*fitting_rules_and_weights)

	relative_fitting_tensor_sets = fitting.build_relative_fitting_tensor_sets(
		*fitting_rules_and_weights
		)

	(bounds_matrix, lowerbounds, upperbounds) = build_bounds(naive)
	inverse_bounds_matrix = np.linalg.pinv(bounds_matrix)

	(init_pars, init_fitness, init_residuals) = build_initial_parameter_values(
		fitting_tensors, relative_fitting_tensor_sets,
		bounds_matrix, lowerbounds, upperbounds
		)

	if np.any(np.abs(init_residuals) > RESIDUAL_CUTOFF):

		print 'Nonzero fitting residuals:'

		print '\n'.join(
			'{} : {:0.2e}'.format(
				entry.id,
				residual
				)
			for entry, residual in zip(fitting_entries, init_residuals)
			if np.abs(residual) > RESIDUAL_CUTOFF
			)

		print 'Overall fit score: {:0.2e}'.format(init_fitness)

	else:
		print 'No nonzero fitting residuals.'

	constraint_penalty_weight = INIT_CONSTRAINT_PENALTY_WEIGHT

	weights = (
		constraint_penalty_weight * OBJ_MASSEQ_WEIGHT,
		constraint_penalty_weight * OBJ_ENERGYEQ_WEIGHT,
		constraint_penalty_weight * OBJ_FLUX_WEIGHT,
		OBJ_FIT_WEIGHT,
		)

	init_obj = ObjectiveValues(
		init_pars, fitting_tensors, relative_fitting_tensor_sets
		).total(*weights)

	table = lo.Table([
		lo.Field('Epoch', 'n'),
		lo.Field('Con. pen.', '.2e', 10),
		lo.Field('Iteration', 'n'),
		lo.Field('Cost', '.3e', 12),
		lo.Field('Fit cost', '.3e', 12),
		])

	log_time = 10

	import time
	last_log_time = time.time()

	best_pars = init_pars.copy()
	best_obj = init_obj

	history_best_objective = []

	perturbation_vectors = build_perturbation_vectors(naive)
	n_perturb = len(perturbation_vectors)

	bounds_range = upperbounds - lowerbounds

	ibm_v = [v for v in inverse_bounds_matrix.T]

	for epoch in xrange(CONSTRAINT_PENALTY_GROWTH_ITERATIONS+1):
		for iteration in xrange(MAX_ITERATIONS):
			deviation = (
				PERTURB_INIT
				* (PERTURB_FINAL/PERTURB_INIT) ** (iteration / MAX_ITERATIONS)
				)

			scale = deviation * random_state.normal()

			dimension = random_state.randint(n_perturb)

			perturbation = scale * perturbation_vectors[dimension]

			new_pars = best_pars + perturbation

			new_acts = bounds_matrix.dot(new_pars)

			unbounded = (lowerbounds > new_acts) | (upperbounds < new_acts)

			for i in np.where(unbounded)[0]:
				bounded = lowerbounds[i] + bounds_range[i] * random_state.random_sample()

				new_pars += (bounded - new_acts[i]) * ibm_v[i]

			new_obj_components = ObjectiveValues(
				new_pars, fitting_tensors, relative_fitting_tensor_sets
				)

			new_obj = new_obj_components.total(*weights)

			did_accept = (new_obj < best_obj)

			history_best_objective.append(
				best_obj
				)

			if did_accept:
				best_obj = new_obj
				best_pars = new_pars

				callback(
					epoch, iteration, constraint_penalty_weight,
					new_obj_components
					)

			log = False
			quit = False

			if iteration == 0:
				log = True

			if time.time() - last_log_time > log_time:
				log = True

			if iteration > MAX_ITERATIONS:
				log = True
				quit = True

			if (iteration >= CONVERGENCE_TIME) and (
					(
						1-best_obj / history_best_objective[-CONVERGENCE_TIME]
						) < CONVERGENCE_RATE
					) and (
						epoch < CONSTRAINT_PENALTY_GROWTH_ITERATIONS
					):
				log = True
				quit = True

			if log:
				table.write(
					epoch,
					constraint_penalty_weight,
					iteration,
					best_obj,
					np.sum(np.abs(fitting_matrix.dot(best_pars) - fitting_values))
					+ compute_relative_fit(best_pars, relative_fitting_tensor_sets)
					)

				last_log_time = time.time()

			if quit:
				break

		constraint_penalty_weight *= CONSTRAINT_PENALTY_GROWTH_RATE

		weights = (
			constraint_penalty_weight * OBJ_MASSEQ_WEIGHT,
			constraint_penalty_weight * OBJ_ENERGYEQ_WEIGHT,
			constraint_penalty_weight * OBJ_FLUX_WEIGHT,
			OBJ_FIT_WEIGHT,
			)

		# must re-evaluate the objective, since the weights changed
		# TODO: store the values instead of the weighted sum
		best_obj = ObjectiveValues(
				best_pars, fitting_tensors, relative_fitting_tensor_sets
				).total(*weights)

	final_pars = best_pars
	final_obj = ObjectiveValues(final_pars, fitting_tensors, relative_fitting_tensor_sets)

	# np.save('temp.npy', np.array(temp))

	return final_pars, final_obj

if __name__ == '__main__':
	(pars, obj) = estimate_parameters(random_state = np.random.RandomState(0))
