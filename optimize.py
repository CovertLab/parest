
from __future__ import division

from itertools import izip

import time

import numpy as np

import liveout as lo

import structure
import fitting
import equations
import utils.linalg as la
import utils.vectorops as vo

from initialization import build_initial_parameter_values

# TODO: some way to pass these as optional arguments

DISEQU_WEIGHTS = (
	np.logspace(-5, +15, 41)
	)

MAX_ITERATIONS = int(1e6) # maximum number of iteration steps per epoch

PERTURBATION_SCALE = np.logspace(
	+2,
	-6,
	MAX_ITERATIONS
	)

CONVERGENCE_RATE = ( # if the objective fails to improve at this rate, assume convergence and move on to the next step
		1e-4 # default
		# 0 # force all iterations - very slow
		)
CONVERGENCE_TIME = int(1e4) # number of iterations between checks to compare - should probably scale with problem size

FORCE_BETTER_INIT = True # use bounds for parsimonious perturbations for initialization, regardless of perturbation approach

TARGET_PYRUVATE_PRODUCTION = 0.14e-3 # the target rate at which the system produces pyruvate, in M/s

LOG_TIME = 10.0 # max time, in seconds, between logging events

TABLE_FIELDS = (
	lo.Field('Epoch', 'n'),
	lo.Field('Con. pen.', '.2e', 10),
	lo.Field('Iteration', 'n'),
	lo.Field('Total error', '.3e', 12),
	lo.Field('Misfit error', '.3e', 12),
	lo.Field('UnSS error', '.3e', 10),
	)

# These options are just for demonstration purposes (enhanced speed vs.
# default numpy operations).  Using my custom functions significantly improves
# optimization times (roughly half of the overall time utilized by the
# numpy-based composed operations).  I attribute these gains to 1) reduced
# overhead and 2) benefits of BLAS (or similar) accelerated dot products.

USE_CUSTOM_FUNCTIONS = True # uses several custom operations designed and optimized for short vectors
USE_NORMS = False # if USE_CUSTOM_FUNCTIONS is False, will try to use norms instead of composed operations (norms are slightly slower)

def fast_square_singleton(x):
	'''
	Probably faster than alternatives.  Apparent gains are minimal.
	'''
	return x*x

if USE_CUSTOM_FUNCTIONS:
	square_singleton = fast_square_singleton
	median1d = vo.fast_shortarray_median1d
	# median1d = vo.fast_shortarray_median1d_partition # alternative; scales better but worse for small vectors
	sumabs1d = vo.fast_shortarray_sumabs1d
	sumsq1d = vo.fast_shortarray_sumsq1d

else:
	square_singleton = np.square
	median1d = np.median

	if USE_NORMS:
		sumabs1d = lambda x: np.linalg.norm(x, 1)
		sumsq1d = lambda x: np.square(np.linalg.norm(x, 2))

	else:
		sumabs1d = lambda x: np.sum(np.abs(x))
		sumsq1d = lambda x: np.sum(np.square(x))

def compute_abs_fit(pars, fitting_tensors):
	(fitting_matrix, fitting_values) = fitting_tensors[:2] # TODO: check overhead of these unpacking operations

	return sumabs1d(fitting_matrix.dot(pars) - fitting_values)

def compute_upper_fit(pars, upper_fitting_tensors):
	(fitting_matrix, fitting_values) = upper_fitting_tensors[:2]

	if fitting_values.size:
		# TODO: consider optimizing sum-fmax operation e.g. (x > 0).dot(x)
		return np.sum(np.fmax(fitting_matrix.dot(pars) - fitting_values, 0.0))

	else:
		# In the standard problem we don't use these penalty functions, so this
		# allows us to skip several expensive (but empty) operations
		return 0.0

def compute_relative_fit(pars, relative_fitting_tensor_sets):
	cost = 0.0

	for fm, fv, fe in relative_fitting_tensor_sets:
		predicted = fm.dot(pars) # TODO: combine dot products into one operation?  might speed up
		d = predicted - fv

		m = median1d(d)

		cost += sumabs1d(d - m)

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

		self.mass_eq = sumsq1d(structure.dynamic_molar_masses * dc_dt)
		self.energy_eq = sumsq1d(dglc_dt)

		net_pyruvate_production = v[-2] - v[-1]

		self.flux = square_singleton(net_pyruvate_production / TARGET_PYRUVATE_PRODUCTION - 1.0)

		self.fit = compute_overall_fit(
			pars,
			fitting_tensors,
			upper_fitting_tensors,
			relative_fitting_tensor_sets
			)

	def misfit_error(self):
		return self.fit

	def disequilibrium_error(self): # TODO: rename as unsteadystate_error
		return self.mass_eq + self.energy_eq + self.flux

	def total(self, disequ_weight):
		return self.misfit_error() + disequ_weight * self.disequilibrium_error()

def nonredundant_vectors(vectors, tolerance = 1e-15):
	# Discards vectors that are equivalent to subsequent vectors under scaling.

	retained = []

	normed = [v / np.sqrt(sumsq1d(v)) for v in vectors]

	# TODO: refactor as a matrix-matrix product?
	for i, vi in enumerate(normed):
		for vj in normed[i+1:]:
			if 1 - np.abs(vi.dot(vj)) < tolerance:
				break

		else:
			retained.append(vectors[i])

	return retained

def build_perturbation_vectors(naive = False):
	if naive:
		# The standard parameter matrix corresponds to the 'usual'
		# parameterization of a kinetic model, in terms of kcat's, metabolite
		# and enzyme concentrations, saturation constants (KM's), and Gibbs
		# standard energies of formation.

		# These perturbations are 'simple' in the sense that a perturbation to
		# a given 'standard' parameter does not influence the value of any
		# other 'standard' parameter.

		perturbation_vectors = list(
			np.linalg.pinv(structure.standard_parameter_matrix).T.copy()
			)

		# TODO: assert perturbations are simple

	else:
		perturbation_vectors = list(
			la.bilevel_elementwise_pseudoinverse(
				np.concatenate([
					structure.activity_matrix,
					np.identity(structure.n_parameters),
					]),
				structure.activity_matrix
				).T.copy()
			)

		# Many of the basic parameter perturbation vectors end up being
		# identical to the 'activity matrix' perturbation vectors, so I detect
		# and strip those from the set.

		perturbation_vectors = nonredundant_vectors(perturbation_vectors)

	return np.array(perturbation_vectors)

import bounds

def build_bounds(naive = False):
	if naive:
		# TODO: move this to bounds.py
		import constants

		lower_conc = bounds.LOWER_CONC
		upper_conc = bounds.UPPER_CONC

		lower_enz_conc = 1e-10 # avg 1/10 molecules per cell
		upper_enz_conc = 1e-3 # most abundant protein is about 750 copies per cell, 1 molecule per E. coli cell ~ 1 nM

		# average protein abundance ~ 1 uM

		lower_kcat = 1e-2 # gives average kcat of about 100 w/ upper kcat of 1e6
		upper_kcat = 1e6 # catalase is around 1e5 /s

		lower_KM = lower_conc
		upper_KM = upper_conc

		lower_glc = constants.RT * np.log(lower_conc)
		upper_glc = constants.RT * np.log(upper_conc)

		lower_gelc = constants.RT * np.log(lower_enz_conc)
		upper_gelc = constants.RT * np.log(upper_enz_conc)

		lower_log_kcat = -constants.RT * np.log(upper_kcat / constants.K_STAR)
		upper_log_kcat = -constants.RT * np.log(lower_kcat / constants.K_STAR)

		lower_log_KM = -constants.RT * np.log(upper_KM)
		upper_log_KM = -constants.RT * np.log(lower_KM)


		bounds_matrix = np.concatenate([
			structure.full_glc_association_matrix,
			structure.gelc_association_matrix,
			structure.kcat_f_matrix,
			# structure.kcat_r_matrix,
			structure.KM_f_matrix,
			structure.KM_r_matrix,
			# structure.Keq_matrix
			])

		lowerbounds = np.concatenate([
			[lower_glc] * structure.full_glc_association_matrix.shape[0],
			[lower_gelc] * structure.gelc_association_matrix.shape[0],
			[lower_log_kcat] * structure.kcat_f_matrix.shape[0],
			# [lower_log_kcat] * structure.kcat_r_matrix.shape[0],
			[lower_log_KM] * structure.KM_f_matrix.shape[0],
			[lower_log_KM] * structure.KM_r_matrix.shape[0],
			# [lower_log_Keq] * structure.Keq_matrix.shape[0],
			])

		upperbounds = np.concatenate([
			[upper_glc] * structure.full_glc_association_matrix.shape[0],
			[upper_gelc] * structure.gelc_association_matrix.shape[0],
			[upper_log_kcat] * structure.kcat_f_matrix.shape[0],
			# [upper_log_kcat] * structure.kcat_r_matrix.shape[0],
			[upper_log_KM] * structure.KM_f_matrix.shape[0],
			[upper_log_KM] * structure.KM_r_matrix.shape[0],
			# [upper_log_Keq] * structure.Keq_matrix.shape[0],
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
		bounds_matrix = bounds.BOUNDS_MATRIX
		lowerbounds = bounds.LOWERBOUNDS
		upperbounds = bounds.UPPERBOUNDS

	return bounds_matrix, lowerbounds, upperbounds

def seconds_to_hms(t):
	hours = t//3600
	minutes = t//60 - 60*hours
	seconds = t - 60*minutes - 3600*hours

	return (hours, minutes, seconds)

def empty_callback(epoch, iteration, constraint_penalty_weight, obj_components):
	pass

def estimate_parameters(
		fitting_rules_and_weights = tuple(),
		random_state = None,
		naive = False,
		random_direction = False,
		callback = empty_callback
		):

	# TODO: pass initial parameters, bounds, perturbation vectors
	# TODO: pass metaparameters
	# TODO: more results/options/callbacks
	# TODO: logging as callbacks

	print 'Initializing optimization.'

	time_start = time.time() # TODO: timing convenience classes

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

	# TODO: if the bounds become more sophisticated, use the calculation for
	# the bilevel elementwise pseudoinverse to acquire the inverse bounds
	# reprojection vectors.  right now it isequivalent to the normal
	# pseudoinverse

	if FORCE_BETTER_INIT:
		init_lowerbounds = bounds.LOWERBOUNDS
		init_upperbounds = bounds.UPPERBOUNDS
		init_bounds_matrix = bounds.BOUNDS_MATRIX

	else:
		init_lowerbounds = lowerbounds
		init_upperbounds = upperbounds
		init_bounds_matrix = bounds_matrix

	(init_pars, init_fitness) = build_initial_parameter_values(
		init_bounds_matrix, (init_lowerbounds + init_upperbounds)/2.0,
		np.concatenate([-init_bounds_matrix, +init_bounds_matrix]), np.concatenate([-init_lowerbounds, +init_upperbounds]),
		fitting_matrix, fitting_values,
		upper_fitting_matrix, upper_fitting_values,
		*[(fm, fv) for (fm, fv, fe) in relative_fitting_tensor_sets]
		)

	# init_pars = np.load('optimized_pars.npy')

	print 'Initial (minimal) fitness: {:0.2f}'.format(init_fitness)

	init_obj_components = ObjectiveValues(
		init_pars,
		fitting_tensors,
		upper_fitting_tensors,
		relative_fitting_tensor_sets
		)

	table = lo.Table(TABLE_FIELDS)

	last_log_time = time.time()

	best_pars = init_pars.copy()
	best_obj_components = init_obj_components

	perturbation_vectors = build_perturbation_vectors(naive)
	n_perturb = len(perturbation_vectors)

	ibm_v = inverse_bounds_matrix.T.copy() # Copy needed to enforce memory-contiguity

	for (epoch, disequ_weight) in enumerate(DISEQU_WEIGHTS):
		history_best_objective = []

		# Need to re-evaluate the objective at the start of every epoch since the weight changes
		best_obj = best_obj_components.total(disequ_weight)

		for (iteration, deviation) in enumerate(PERTURBATION_SCALE):
			if not random_direction:
				dimension = random_state.randint(n_perturb)
				direction = perturbation_vectors[dimension]

			else:
				# TODO: optimize this optional approach
				coeffs = random_state.normal(size = n_perturb)
				coeffs /= np.sqrt(sumsq1d(coeffs))

				direction = perturbation_vectors.T.dot(coeffs) # dot product on transposed matrix is probably slow

			scale = deviation * random_state.normal()
			perturbation = scale * direction

			new_pars = best_pars + perturbation

			# There are various possible execution strategies for bounding.
			# This setup gives the best performance, probably because
			# 'unbounded' is typically False everywhere.
			new_acts = bounds_matrix.dot(new_pars)
			unbounded = (lowerbounds > new_acts) | (upperbounds < new_acts)

			for i in np.where(unbounded)[0]:
				bounded = random_state.uniform(lowerbounds[i], upperbounds[i])

				new_pars += (bounded - new_acts[i]) * ibm_v[i]

			new_obj_components = ObjectiveValues(
				new_pars,
				fitting_tensors,
				upper_fitting_tensors,
				relative_fitting_tensor_sets
				)

			new_obj = new_obj_components.total(disequ_weight)

			did_accept = (new_obj < best_obj)

			history_best_objective.append(best_obj)

			if did_accept:
				best_obj_components = new_obj_components
				best_obj = new_obj
				best_pars = new_pars

			log = False
			quit = False

			if (iteration == 0) or (iteration == MAX_ITERATIONS-1):
				log = True

			if time.time() - last_log_time > LOG_TIME:
				log = True

			if (iteration >= CONVERGENCE_TIME) and (
					(
						1.0 - best_obj / history_best_objective[-CONVERGENCE_TIME]
						) < CONVERGENCE_RATE
					) and (
						epoch < len(DISEQU_WEIGHTS)-1
					):
				# TODO: instead of ending the epoch, jump ahead to a time where
				# perturbations are smaller
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

	print 'Total optimization time: {:02n}:{:02n}:{:05.2f} (H:M:S)'.format(*seconds_to_hms(time_run))

	# TODO: consider polishing via gradient descent or just be minimizing
	# misfit using an L1 norm subject to constraints on intermediate values,
	# either at the end of the whole optimization or the end of each epoch.

	return (best_pars, best_obj_components)

if __name__ == '__main__':
	import problems
	definition = problems.DEFINITIONS['all_scaled']

	(pars, obj) = estimate_parameters(
		definition,
		random_state = np.random.RandomState(0),
		# naive = True,
		# random_direction = True
		)

	np.save('optimized_pars.npy', pars)
