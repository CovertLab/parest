
from __future__ import division

from itertools import izip

import numpy as np

import liveout as lo

from utils.bounds import unbounded_to_random

import constants
import structure
import fitting
import equations
import utils.linalg as la

from initialization import build_initial_parameter_values
import bounds

MAX_ITERATIONS = int(1e6)
CONVERGENCE_RATE = 1e-4
CONVERGENCE_TIME = int(1e4)

PERTURB_INIT = 1e1
PERTURB_FINAL = 1e-6

OBJ_MASSEQ_WEIGHT = 1e0
OBJ_ENERGYEQ_WEIGHT = 1e0
OBJ_FLUX_WEIGHT = 1e0

INIT_OBJ_FIT_WEIGHT = 1e10
FINAL_OBJ_FIT_WEIGHT = 1e-10
FALLOFF_RATE = 1e-1
FALLOFF_ITERATIONS = int(np.ceil(
	np.log(FINAL_OBJ_FIT_WEIGHT / INIT_OBJ_FIT_WEIGHT) / np.log(FALLOFF_RATE)
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

def unit_basis(index, size, dtype = None):
	vec = np.zeros(size, dtype)
	vec[index] = 1

	return vec

PERTURB_NAIVE = False

if PERTURB_NAIVE:
	perturbation_vectors = [
		unit_basis(i, structure.n_parameters)
		for i in xrange(structure.n_parameters)
		]

else:
	dynamic_conc_indices = {
		structure.parameters.index(
			structure.GLC.format(compound)
			)
		for compound in structure.DYNAMIC_COMPOUNDS
		}

	perturbation_vectors = [
		vector
		for vector in la.bilevel_elementwise_pseudoinverse(
			np.concatenate([
				[
					unit_basis(i, structure.n_parameters)
					for i in xrange(structure.n_parameters)
					if i not in dynamic_conc_indices
					],
				structure.activity_matrix
				]),
			structure.activity_matrix
			).T
		]

n_perturb = len(perturbation_vectors)

FIT_INIT = True
RESIDUAL_CUTOFF = 1e-5

def estimate_parameters(fitting_rules_and_weights = tuple(), random_state = np.random):

	fitting_tensors = (
		fitting_matrix,
		fitting_values,
		fitting_entries
		) = fitting.build_fitting_tensors(*fitting_rules_and_weights)

	relative_fitting_tensor_sets = fitting.build_relative_fitting_tensor_sets(
		*fitting_rules_and_weights
		)

	if FIT_INIT:

		(
			init_pars,
			init_fitness,
			init_residuals
			) = build_initial_parameter_values(fitting_tensors, relative_fitting_tensor_sets)

	else:
		(
			init_pars,
			init_fitness,
			init_residuals
			) = build_initial_parameter_values(np.empty((0, 0)), np.empty((0,)))

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

	obj_fit_weight = INIT_OBJ_FIT_WEIGHT

	weights = (
		OBJ_MASSEQ_WEIGHT,
		OBJ_ENERGYEQ_WEIGHT,
		OBJ_FLUX_WEIGHT,
		obj_fit_weight,
		)

	init_obj = ObjectiveValues(
		init_pars, fitting_tensors, relative_fitting_tensor_sets
		).total(*weights)

	table = lo.Table([
		lo.Field('Step', 'n'),
		lo.Field('Fit weight', '.2e', 10),
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

	for step in xrange(FALLOFF_ITERATIONS):
		for iteration in xrange(MAX_ITERATIONS):
			deviation = (
				PERTURB_INIT
				* (PERTURB_FINAL/PERTURB_INIT) ** (iteration / MAX_ITERATIONS)
				)

			scale = deviation * random_state.normal()

			dimension = random_state.randint(n_perturb)

			new_pars = best_pars + scale * perturbation_vectors[dimension]

			new_acts = bounds.BOUNDS_MATRIX.dot(new_pars)
			bounded_acts = unbounded_to_random(
				new_acts,
				bounds.UPPERBOUNDS, bounds.LOWERBOUNDS,
				random_state
				)

			new_pars += bounds.INVERSE_BOUNDS_MATRIX.dot(bounded_acts - new_acts)
			new_obj = ObjectiveValues(
				new_pars, fitting_tensors, relative_fitting_tensor_sets
				).total(*weights)

			did_accept = (new_obj < best_obj)

			history_best_objective.append(
				best_obj
				)

			if did_accept:
				best_obj = new_obj
				best_pars = new_pars

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
					):
				log = True
				quit = True

			if log:
				table.write(
					step,
					obj_fit_weight,
					iteration,
					best_obj,
					np.sum(np.abs(fitting_matrix.dot(best_pars) - fitting_values))
					+ compute_relative_fit(best_pars, relative_fitting_tensor_sets)
					)

				last_log_time = time.time()

			if quit:
				break

		obj_fit_weight *= FALLOFF_RATE

		weights = (
			OBJ_MASSEQ_WEIGHT,
			OBJ_ENERGYEQ_WEIGHT,
			OBJ_FLUX_WEIGHT,
			obj_fit_weight,
			)

		# must re-evaluate the objective, since the weights changed
		# TODO: store the values instead of the weighted sum
		best_obj = ObjectiveValues(
				best_pars, fitting_tensors, relative_fitting_tensor_sets
				).total(*weights)

	final_pars = best_pars
	final_obj = ObjectiveValues(final_pars, fitting_tensors, relative_fitting_tensor_sets)

	return final_pars, final_obj

if __name__ == '__main__':
	(pars, obj) = estimate_parameters(random_state = np.random.RandomState(0))
