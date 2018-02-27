
from __future__ import division

from itertools import izip

import numpy as np

import liveout as lo

import structure
import fitting
import equations
import utils.linalg as la

from initialization import build_initial_parameter_values

import time

# TODO: some way to pass these as optional arguments

DISEQU_WEIGHTS = np.logspace(-10, +10, 41)

MAX_ITERATIONS = int(1e6) # maximum number of iteration steps per epoch

PERTURBATION_SCALE = np.logspace(+2, -6, MAX_ITERATIONS)

CONVERGENCE_RATE = 1e-4 # if the objective fails to improve at this rate, assume convergence and move on to the next step
CONVERGENCE_TIME = int(1e4) # number of iterations between checks to compare - should probably scale with problem size

TARGET_PYRUVATE_PRODUCTION = 1e-3 # the target rate at which the system produces pyruvate

LOG_TIME = 10

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

def compute_abs_fit(pars, fitting_tensors):
	(fitting_matrix, fitting_values) = fitting_tensors[:2]

	return np.sum(np.abs(fitting_matrix.dot(pars) - fitting_values))

def compute_upper_fit(pars, upper_fitting_tensors):
	(fitting_matrix, fitting_values) = upper_fitting_tensors[:2]

	return np.sum(np.fmax(fitting_matrix.dot(pars) - fitting_values, 0))

def compute_relative_fit(pars, relative_fitting_tensor_sets):
	cost = 0

	for (fm, fv, fe) in relative_fitting_tensor_sets:
		predicted = fm.dot(pars)
		d = predicted - fv

		m = median1d(d)

		cost += np.sum(np.abs(d - m))

	return cost

def compute_overall_fit(pars, fitting_tensors, upper_fitting_tensors, relative_fitting_tensor_sets):
	return (
		compute_abs_fit(pars, fitting_tensors)
		+ compute_upper_fit(pars, upper_fitting_tensors)
		+ compute_relative_fit(pars, relative_fitting_tensor_sets)
		)

class ObjectiveValues(object):
	def __init__(self, pars, fitting_tensors, upper_fitting_tensors, relative_fitting_tensor_sets = ()):
		(v, dc_dt, dglc_dt) = equations.compute_all(pars, *equations.args)

		self.mass_eq = np.sum(np.square(structure.dynamic_molar_masses * dc_dt))
		self.energy_eq = np.sum(np.square(dglc_dt))

		net_pyruvate_production = v[-2] - v[-1]

		self.flux = (net_pyruvate_production / TARGET_PYRUVATE_PRODUCTION - 1)**2

		self.fit = compute_overall_fit(pars, fitting_tensors, upper_fitting_tensors, relative_fitting_tensor_sets)

	def misfit_error(self):
		return self.fit

	def disequilibrium_error(self):
		return self.mass_eq + self.energy_eq + self.flux

	def total(self, disequ_weight):
		return self.misfit_error() + disequ_weight * self.disequilibrium_error()


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

	return np.array(perturbation_vectors)

def build_bounds(naive = False):
	if naive:
		# TODO: move this to bounds.py
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
			# structure.kcat_r_matrix,
			structure.KM_f_matrix,
			structure.KM_r_matrix,
			])

		lowerbounds = np.concatenate([
			[lower_glc] * structure.full_glc_association_matrix.shape[0],
			[lower_gelc] * structure.gelc_association_matrix.shape[0],
			[lower_log_kcat] * structure.kcat_f_matrix.shape[0],
			# [lower_log_kcat] * structure.kcat_r_matrix.shape[0],
			[lower_log_KM] * structure.KM_f_matrix.shape[0],
			[lower_log_KM] * structure.KM_r_matrix.shape[0],
			])

		upperbounds = np.concatenate([
			[upper_glc] * structure.full_glc_association_matrix.shape[0],
			[upper_gelc] * structure.gelc_association_matrix.shape[0],
			[upper_log_kcat] * structure.kcat_f_matrix.shape[0],
			# [upper_log_kcat] * structure.kcat_r_matrix.shape[0],
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

def empty_callback(epoch, iteration, constraint_penalty_weight, obj_components):
	pass

def estimate_parameters(
		fitting_rules_and_weights = tuple(),
		random_state = None,
		naive = False,
		force_better_init = False,
		random_direction = False,
		callback = empty_callback
		):

	print 'Initializing optimization.'

	time_start = time.time()

	if random_state is None:
		print 'No random state provided.'
		random_state = np.random.RandomState()

	fitting_tensors = (
		fitting_matrix,
		fitting_values,
		fitting_entries
		) = fitting.build_fitting_tensors(*fitting_rules_and_weights)

	upper_fitting_tensors = (
		upper_fitting_matrix,
		upper_fitting_values,
		upper_fitting_entries
		) = fitting.build_upper_fitting_tensors(*fitting_rules_and_weights)

	relative_fitting_tensor_sets = fitting.build_relative_fitting_tensor_sets(
		*fitting_rules_and_weights
		)

	(bounds_matrix, lowerbounds, upperbounds) = build_bounds(naive)
	inverse_bounds_matrix = np.linalg.pinv(bounds_matrix)

	if not force_better_init:
		init_bounds_matrix = bounds_matrix
		init_lowerbounds = lowerbounds
		init_upperbounds = upperbounds

	else:
		# The naive parameter bounds do a poor job of regularizing the
		# ranges of the initial values.  Subsequently the initial
		# parameters have an extremely large associated disequilibrium
		# error.  Using this option leads to 'fairer' initialization,
		# in the sense that both 'naive' and 'parsimonious' start from
		# the same position.
		(
			init_bounds_matrix,
			init_lowerbounds,
			init_upperbounds
			) = build_bounds(naive = False)

	(init_pars, init_fitness) = build_initial_parameter_values(
		init_bounds_matrix, (init_lowerbounds + init_upperbounds)/2,
		np.concatenate([-init_bounds_matrix, +init_bounds_matrix]), np.concatenate([-init_lowerbounds, init_upperbounds]),
		fitting_matrix, fitting_values,
		upper_fitting_matrix, upper_fitting_values,
		*[(fm, fv) for (fm, fv, fe) in relative_fitting_tensor_sets]
		)

	print 'Initial (minimal) fitness: {:0.2f}'.format(init_fitness)

	init_obj_components = ObjectiveValues(
		init_pars,
		fitting_tensors,
		upper_fitting_tensors,
		relative_fitting_tensor_sets
		)

	table = lo.Table([
		lo.Field('Epoch', 'n'),
		lo.Field('Con. pen.', '.2e', 10),
		lo.Field('Iteration', 'n'),
		lo.Field('Total error', '.3e', 12),
		lo.Field('Misfit error', '.3e', 12),
		lo.Field('Diseq. error', '.3e', 12),
		])

	last_log_time = time.time()

	best_pars = init_pars.copy()
	best_obj_components = init_obj_components

	history_best_objective = []

	perturbation_vectors = build_perturbation_vectors(naive)
	n_perturb = len(perturbation_vectors)

	bounds_range = upperbounds - lowerbounds

	ibm_v = [v for v in inverse_bounds_matrix.T]

	for (epoch, disequ_weight) in enumerate(DISEQU_WEIGHTS):
		# Re-evaluate the objective, since the weight changes
		best_obj = best_obj_components.total(disequ_weight)

		for (iteration, deviation) in enumerate(PERTURBATION_SCALE):
			scale = deviation * random_state.normal()

			if not random_direction:
				dimension = random_state.randint(n_perturb)
				direction = perturbation_vectors[dimension]

			else:
				coeffs = np.random.normal(size = n_perturb)
				coeffs /= np.sqrt(np.sum(np.square(coeffs)))

				direction = perturbation_vectors.T.dot(coeffs) # dot product on transposed matrix is probably slow

			perturbation = scale * direction

			new_pars = best_pars + perturbation

			new_acts = bounds_matrix.dot(new_pars)

			unbounded = (lowerbounds > new_acts) | (upperbounds < new_acts)

			for i in np.where(unbounded)[0]:
				bounded = lowerbounds[i] + bounds_range[i] * random_state.random_sample()

				new_pars += (bounded - new_acts[i]) * ibm_v[i]

			new_obj_components = ObjectiveValues(
				new_pars,
				fitting_tensors,
				upper_fitting_tensors,
				relative_fitting_tensor_sets
				)

			new_obj = new_obj_components.total(disequ_weight)

			did_accept = (new_obj < best_obj)

			history_best_objective.append(
				best_obj
				)

			if did_accept:
				best_obj_components = new_obj_components
				best_obj = new_obj
				best_pars = new_pars

			log = False
			quit = False

			if iteration == 0:
				log = True

			if time.time() - last_log_time > LOG_TIME:
				log = True

			if (iteration >= CONVERGENCE_TIME) and (
					(
						1-best_obj / history_best_objective[-CONVERGENCE_TIME]
						) < CONVERGENCE_RATE
					) and (
						epoch < len(DISEQU_WEIGHTS)-1
					):
				log = True
				quit = True

			if log:
				table.write(
					epoch,
					disequ_weight,
					iteration,
					best_obj,
					best_obj_components.misfit_error(),
					best_obj_components.disequilibrium_error()
					)

				last_log_time = time.time()

			if log or did_accept:
				callback(
					epoch, iteration, disequ_weight,
					new_obj_components
					)

			if quit:
				break

	time_run = time.time() - time_start

	hours = time_run//3600
	minutes = time_run//60 - 60*hours
	seconds = time_run - 60*minutes - 3600*hours

	print 'Completed in {:02n}:{:02n}:{:0.2f} (H:M:S)'.format(hours, minutes, seconds)

	return best_pars, best_obj_components

if __name__ == '__main__':
	import problems
	definition = problems.DEFINITIONS['all_scaled']

	(pars, obj) = estimate_parameters(
		definition,
		random_state = np.random.RandomState(0),
		# naive = True,
		# force_better_init = True,
		# random_direction = True
		)

	np.save('optimized_pars.npy', pars)
