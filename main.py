
from __future__ import division

from itertools import izip

from os.path import join

import numpy as np
# import matplotlib.pyplot as plt

import liveout as lo

from utils.bounds import unbounded_to_random

from optimize import optimize

import constants
import structure
import fitting
import equations
import utils.linalg as la
from utils.rdp import rdp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type = int, default = None)

args = parser.parse_args()

np.random.seed(args.seed)

RESOLUTION = np.finfo(np.float64).resolution

log_every = 1e5

perturb_activities = True

max_iterations = int(1e6)
convergence_rate = 1e-4
convergence_time = int(1e4)

perturb_init = 1e1
perturb_final = 1e-6 # might need to tune down

obj_abseq_weight = 1e0
obj_releq_weight = 1e0
obj_flux_weight = 1e0 # might need to bump up

init_obj_fit_weight = 1e10 # choose based on initial eq score
final_obj_fit_weight = 1e-10
falloff_rate = 1e-1
falloff_iterations = int(np.ceil(
	np.log(final_obj_fit_weight / init_obj_fit_weight) / np.log(falloff_rate)
	)) # continue to fall off until constraint penalties are sufficiently small

target_pyruvate_production = 1e-3

fitting_rules_and_weights = (
	(lambda entry: True, 1.0),
	)

activity_matrix = np.concatenate([
	structure.forward_saturated_reaction_potential_matrix,
	structure.reverse_saturated_reaction_potential_matrix,
	structure.solo_forward_binding_potential_matrix,
	structure.solo_reverse_binding_potential_matrix,
	structure.glc_association_matrix,
	# structure.full_glc_association_matrix,
	# structure.gelc_association_matrix
	])

if perturb_activities:
	perturbation_matrix = activity_matrix

	basal_vmax = 1e-5 # M/s
	basal_saturation_ratio = 1 # dimensionless, C / KM
	basal_c = 1e-2 # M, mu * C should be ~ vMax

	basal_rp = constants.RT * np.log(basal_vmax / constants.K_STAR)
	basal_bp = constants.RT * np.log(basal_saturation_ratio)
	basal_glc = basal_gelc = constants.RT * np.log(basal_c)

	bounds_rp = (
		constants.RT * np.log(basal_vmax * np.sqrt(RESOLUTION)),
		constants.RT * np.log(basal_vmax / np.sqrt(RESOLUTION)),
		)
	bounds_bp = (
		constants.RT * np.log(1*np.sqrt(RESOLUTION)),
		constants.RT * np.log(1/np.sqrt(RESOLUTION)),
		)
	bounds_glc = (
		constants.RT * np.log(basal_c * np.sqrt(RESOLUTION)),
		constants.RT * np.log(basal_c / np.sqrt(RESOLUTION)),
		)
	# bounds_gelc = (
	# 	constants.RT * np.log(basal_c * np.sqrt(RESOLUTION)),
	# 	constants.RT * np.log(basal_c / np.sqrt(RESOLUTION)),
	# 	)

	bounds = (
		[bounds_rp] * (
			structure.forward_saturated_reaction_potential_matrix.shape[0]
			+ structure.reverse_saturated_reaction_potential_matrix.shape[0]
			)
		+ [bounds_bp] * (
			structure.solo_forward_binding_potential_matrix.shape[0]
			+ structure.solo_reverse_binding_potential_matrix.shape[0]
			)
		+ [bounds_glc] * structure.glc_association_matrix.shape[0]
		# + [bounds_glc] * structure.full_glc_association_matrix.shape[0]
		# + [bounds_gelc] * structure.gelc_association_matrix.shape[0]
		)

else:
	# TODO: instead of perturbing the raw parameters, transform into
	# more familiar parameters
	# i.e. log k_cat, log KM, log Keq, log C, log E
	# and bound; should add a bunch of new matrices to structure.py

	perturbation_matrix = np.identity(structure.n_parameters)

	raise NotImplementedError()

(n_perturb, n_pars) = perturbation_matrix.shape

inverse_perturbation_matrix = la.pinv(perturbation_matrix)

(lowerbounds, upperbounds) = np.column_stack(bounds)

(fitting_matrix, fitting_values, fitting_ids) = fitting.build_fitting_tensors(
	*fitting_rules_and_weights
	)

def build_initial_parameter_values():
	init_perturb = (lowerbounds + upperbounds)/2
	init_perturb += 0.1 * (upperbounds - lowerbounds) * np.random.normal(size = n_perturb)
	# init_perturb += (upperbounds - lowerbounds)/4 * np.random.normal(size = n_perturb)

	# init_perturb = np.random.random(n_perturb) * (upperbounds - lowerbounds) + lowerbounds

	from utils.l1min import linear_least_l1_regression

	A = fitting_matrix
	b = fitting_values

	G = np.concatenate([
		-perturbation_matrix,
		+perturbation_matrix
		])

	h = np.concatenate([
		-lowerbounds,
		upperbounds
		]) - 1e-6 # tighten the bounds slightly
	# I believe this to be an issue with the precision of the solver

	fit_pars, fitness = linear_least_l1_regression(A, b, G, h)

	assert (
		(perturbation_matrix.dot(fit_pars) >= lowerbounds)
		& (perturbation_matrix.dot(fit_pars) <= upperbounds)
		).all(), 'fit parameters not within bounds'

	N = la.nullspace_projector(fitting_matrix)

	from scipy.optimize import minimize

	res = minimize(
		lambda z: np.sum(np.square(
			perturbation_matrix.dot(fit_pars + N.dot(z)) - init_perturb
			)),
		np.zeros(N.shape[1]),
		constraints = dict(
			type = 'ineq',
			fun = lambda z: h - G.dot(fit_pars + N.dot(z))
			)
		)

	assert res.success

	z = res.x

	init_pars = fit_pars + N.dot(z)

	assert (
		(perturbation_matrix.dot(init_pars) >= lowerbounds)
		& (perturbation_matrix.dot(init_pars) <= upperbounds)
		).all(), 'init parameters not within bounds'

	assert np.abs(
		fitness
		- np.sum(np.abs(fitting_matrix.dot(init_pars) - fitting_values))
		) < 1e-10, 'init parameters not fit'

	return init_pars

def obj_abseq(dc_dt):
	return np.sum(np.square(structure.dynamic_molar_masses * dc_dt))

def obj_releq(dglc_dt):
	return np.sum(np.square(dglc_dt))

def obj_flux(v):
	net_pyruvate_production = v[-2] - v[-1]

	return (net_pyruvate_production / target_pyruvate_production - 1)**2

def obj_fit(x):
	return np.sum(np.abs(fitting_matrix.dot(x) - fitting_values))

def objective(x):
	(v, dc_dt, dglc_dt) = equations.compute_all(x, *equations.args)

	return (
		obj_abseq_weight * obj_abseq(dc_dt)
		+ obj_releq_weight * obj_releq(dglc_dt)
		+ obj_flux_weight * obj_flux(v)
		+ obj_fit_weight * obj_fit(x)
		)

def perturbation_function(optimization_result):
	old_x = optimization_result.x
	iteration = optimization_result.i

	deviation = (
		perturb_init
		* (perturb_final/perturb_init) ** (iteration / max_iterations)
		)

	dimension = np.random.randint(n_perturb)

	old_acts = perturbation_matrix.dot(old_x)

	new_acts = old_acts.copy()

	new_acts[dimension] += deviation * np.random.normal()

	bounded_acts = unbounded_to_random(new_acts, upperbounds, lowerbounds)

	new_x = old_x + inverse_perturbation_matrix.dot(
		bounded_acts - old_acts
		)

	return new_x

table = lo.Table([
	lo.Field('Step', 'n'),
	lo.Field('Fit weight', '.2e', 10),
	lo.Field('Iteration', 'n'),
	lo.Field('Cost', '.2e', 10),
	lo.Field('Fit cost', '.2e', 10),
	])

log_time = 10

import time
last_log_time = time.time()

init_pars = build_initial_parameter_values()

obj_fit_weight = init_obj_fit_weight

pars = init_pars.copy()

for step in xrange(falloff_iterations):

	history_f = []

	for iterate in optimize(
			pars,
			objective,
			perturbation_function
			):

		history_f.append(
			iterate.best.f
			)

		log = False
		quit = False

		# if (iterate.iteration%int(log_every)) == 0:
		# 	log = True

		if iterate.iteration == 0:
			log = True

		if time.time() - last_log_time > log_time:
			log = True

		if iterate.iteration > max_iterations:
			log = True
			quit = True

		if (iterate.iteration >= convergence_time) and (
				(
					1-iterate.best.f / history_f[-convergence_time]
					) < convergence_rate
				):
			log = True
			quit = True

		if log:
			table.write(
				step,
				obj_fit_weight,
				iterate.iteration,
				iterate.best.f,
				np.sum(np.abs(fitting_matrix.dot(iterate.best.x) - fitting_values))
				)

			last_log_time = time.time()

		if quit:
			break

	pars = iterate.best.x
	obj_fit_weight *= falloff_rate

final_pars = iterate.best.x

v = equations.reaction_rates(final_pars, *equations.args)

final_mass_eq = np.sum(np.square(structure.dynamic_molar_masses * equations.dc_dt(final_pars, *equations.args)))
final_energy_eq = np.sum(np.square(equations.dglc_dt(final_pars, *equations.args)))
final_flux = ((v[-2] - v[-1])/target_pyruvate_production - 1)**2
final_fit = np.sum(np.abs(fitting_matrix.dot(final_pars) - fitting_values))

print final_mass_eq, final_energy_eq, final_flux, final_fit

if args.seed is not None:
	np.save('out/ip{}.npy'.format(args.seed), init_pars)
	np.save('out/fp{}.npy'.format(args.seed), final_pars)
	np.save('out/s{}.npy'.format(args.seed), np.array([final_mass_eq, final_energy_eq, final_flux, final_fit]))
