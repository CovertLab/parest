
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

RESOLUTION = np.finfo(np.float64).resolution

max_iterations = int(1e6)
convergence_rate = 1e-4
convergence_time = int(1e4)

perturb_init = 1e1
perturb_final = 1e-10

obj_abseq_weight = 1e6
obj_releq_weight = 1e0
obj_flux_weight = obj_abseq_weight
obj_fit_weight = 1e-6

target_pyruvate_production = 1e-3

fit_gs_weight = 1e-2
fit_glc_weight = 1e0
fit_kcat_weight = 1e-1
fit_km_weight = 1e-1

activity_matrix = np.concatenate([
	structure.forward_saturated_reaction_potential_matrix,
	structure.reverse_saturated_reaction_potential_matrix,
	structure.solo_reverse_binding_potential_matrix,
	structure.solo_forward_binding_potential_matrix,
	structure.full_glc_association_matrix,
	structure.gelc_association_matrix
	])

init_vmax = 1e-6 # M/s
init_c = 1e-6 # M

init_rp = constants.RT * np.log(init_vmax / constants.K_STAR)
init_bp = 0 # C = K_M
init_glc = init_gelc = constants.RT * np.log(init_c)

init_acts = np.concatenate([
	[init_rp] * structure.forward_saturated_reaction_potential_matrix.shape[0]
	+ [init_rp] * structure.reverse_saturated_reaction_potential_matrix.shape[0]
	+ [init_bp] * structure.solo_reverse_binding_potential_matrix.shape[0]
	+ [init_bp] * structure.solo_forward_binding_potential_matrix.shape[0]
	+ [init_glc] * structure.full_glc_association_matrix.shape[0]
	+ [init_gelc] * structure.gelc_association_matrix.shape[0]
	])

(n_acts, n_pars) = activity_matrix.shape

inverse_activity_matrix = la.pinv(activity_matrix)

bounds_rp = (
	constants.RT * np.log(init_vmax * np.sqrt(RESOLUTION)),
	constants.RT * np.log(init_vmax / np.sqrt(RESOLUTION)),
	)
bounds_bp = (
	constants.RT * np.log(1*np.sqrt(RESOLUTION)),
	constants.RT * np.log(1/np.sqrt(RESOLUTION)),
	)
bounds_glc = (
	constants.RT * np.log(init_c * np.sqrt(RESOLUTION)),
	constants.RT * np.log(init_c / np.sqrt(RESOLUTION)),
	)
bounds_gelc = (
	constants.RT * np.log(init_c * np.sqrt(RESOLUTION)),
	constants.RT * np.log(init_c / np.sqrt(RESOLUTION)),
	)

bounds = (
	[bounds_rp] * structure.forward_saturated_reaction_potential_matrix.shape[0]
	+ [bounds_rp] * structure.reverse_saturated_reaction_potential_matrix.shape[0]
	+ [bounds_bp] * structure.solo_reverse_binding_potential_matrix.shape[0]
	+ [bounds_bp] * structure.solo_forward_binding_potential_matrix.shape[0]
	+ [bounds_glc] * structure.full_glc_association_matrix.shape[0]
	+ [bounds_gelc] * structure.gelc_association_matrix.shape[0]
	)

(lowerbounds, upperbounds) = zip(*bounds)

lowerbounds = np.asarray(lowerbounds)
upperbounds = np.asarray(upperbounds)

def build_fitting_tensors():
	fw = [
		fit_gs_weight,
		fit_glc_weight,
		fit_kcat_weight, # forward
		fit_kcat_weight, # reverse
		fit_km_weight,
		]

	fm = [
		fitting.gs_mat,
		fitting.glc_mat,
		fitting.kf_mat,
		fitting.kr_mat,
		fitting.KM_mat,
		]

	fv = [
		fitting.gs_values,
		fitting.glc_values,
		fitting.kf_values,
		fitting.kr_values,
		fitting.KM_values,
		]

	fitting_matrix = []
	fitting_values = []

	for w, m, v in izip(fw, fm, fv):
		if w == 0:
			continue

		else:
			fitting_matrix.append(w*m)
			fitting_values.append(w*v)

	fitting_matrix = np.concatenate(fitting_matrix)
	fitting_values = np.concatenate(fitting_values)

	return (fitting_matrix, fitting_values)

(fitting_matrix, fitting_values) = build_fitting_tensors()

def build_initial_parameter_values():
	from utils.l1min import linear_least_l1_regression

	# TODO: validate these choices of weights
	w_fit = 1e0
	w_act = 1e-6
	w_reg = 1e-12

	init_pars = linear_least_l1_regression(
		np.concatenate([
			w_fit * fitting_matrix,
			w_act * activity_matrix,
			w_reg * np.identity(n_pars)
			]),
		np.concatenate([
			w_fit * fitting_values,
			w_act * init_acts,
			w_reg * np.zeros(n_pars)
			])
		)[0]

	return init_pars

init_pars = build_initial_parameter_values()

def obj_abseq(dc_dt):
	return np.sum(np.square(dc_dt))

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

	dimension = np.random.randint(n_acts)

	old_acts = activity_matrix.dot(old_x)

	new_acts = old_acts.copy()

	new_acts[dimension] += deviation * np.random.normal()

	bounded_acts = unbounded_to_random(new_acts, upperbounds, lowerbounds)

	new_x = old_x + inverse_activity_matrix.dot(
		bounded_acts - old_acts
		)

	return new_x

table = lo.Table([
	lo.Field('Iteration', 'n'),
	lo.Field('Cost', '.2e', 10),
	])

history_f = []

for iterate in optimize(
		init_pars,
		objective,
		perturbation_function
		):

	if (iterate.iteration%int(1e3)) == 0:
		table.write(
			iterate.iteration,
			iterate.best.f
			)

	history_f.append(
		iterate.best.f
		)

	if iterate.iteration > max_iterations:
		break

	if (iterate.iteration > convergence_time) and (
			(1-iterate.best.f / history_f[-convergence_time]) < convergence_rate
			):
		break

import matplotlib.pyplot as plt

f = np.array(history_f)
i = np.arange(f.size)

points = np.column_stack([i, f])

mask = rdp(points, 1e-3)

plt.semilogy(i[mask], f[mask], 'k-')

plt.show()

import ipdb; ipdb.set_trace()
