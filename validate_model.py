
from __future__ import division

import os.path as pa
from itertools import izip

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

import problems
import constants
import equations
import structure
from utils.linalg import approx_jac

CONC_FOLD_PERTURBATION = 2
CONC_FOLD_CONVERGENCE = 1.01

PERTURBATION_WIDTH = constants.RT * np.log(CONC_FOLD_PERTURBATION)
N_PERTURBATIONS = 10

APPROX_JAC_RADIUS = 1e-3

PERTURBATION_RECOVERTY_TIME_TOLERANCE = 2
EXPECTED_RECOVERY_EPOCHS = np.log((CONC_FOLD_PERTURBATION - 1)/(CONC_FOLD_CONVERGENCE - 1))
PERTURBATION_RECOVERY_EPOCHS = PERTURBATION_RECOVERTY_TIME_TOLERANCE * EXPECTED_RECOVERY_EPOCHS

DT = 1e-1
T_INIT = 0

def load_pars(problem):
	obj = np.load(pa.join('out', problem, 'obj.npy'))
	pars = np.load(pa.join('out', problem, 'pars.npy'))

	return obj, pars

conc_ind = [
	structure.parameters.index(structure.GLC.format(compound))
	for compound in structure.DYNAMIC_COMPOUNDS
	]
is_dynamic = np.zeros(structure.n_parameters, np.bool)
is_dynamic[conc_ind] = True
is_static = ~is_dynamic

def dg_dt(glc, pars):
	x = structure.glc_association_matrix.T.dot(glc)
	x[is_static] = pars[is_static]

	return equations.dglc_dt(x, *equations.args)

def init_dg_dt(pars):
	return structure.glc_association_matrix.dot(pars)

def has_recovered(x, x_eq):
	return np.all(np.abs(x - x_eq) < constants.RT * np.log(CONC_FOLD_CONVERGENCE))

temp = 0 #18

for problem in ['data_agnostic']: #problems.DEFINITIONS.viewkeys():
	(all_obj, all_pars) = load_pars(problem)

	lre = []
	stable = []

	print all_obj[:, temp]

	for pars in [all_pars[:, temp]]: #all_pars.T:

		dx_dt = lambda t, x: dg_dt(x, pars)

		x_start = init_dg_dt(pars)

		t_final = PERTURBATION_RECOVERY_EPOCHS / constants.MU

		ode = scipy.integrate.ode(dx_dt)

		ode.set_initial_value(x_start, T_INIT)

		while ode.successful() and ode.t < t_final:
			x_curr = ode.integrate(ode.t + DT)
			# TODO: terminate when x_curr stops changing

		if not ode.successful():
			stable.append(False)
			continue

		x_eq = x_curr

		# x_eq = init_dg_dt(pars)

		largest_real_eigenvalue = np.max(np.real(np.linalg.eigvals(
			approx_jac(lambda x: dx_dt(T_INIT, x), x_eq, APPROX_JAC_RADIUS)
			)))

		print largest_real_eigenvalue / constants.MU

		lre.append(largest_real_eigenvalue)

		if largest_real_eigenvalue >= 0:
			stable.append(False)
			continue

		t_final = -PERTURBATION_RECOVERY_EPOCHS / largest_real_eigenvalue

		x_final = []

		for p in xrange(N_PERTURBATIONS):
			x_init = x_eq + (np.random.uniform(size = x_eq.size) - 0.5) * PERTURBATION_WIDTH

			ode = scipy.integrate.ode(dx_dt)

			ode.set_initial_value(x_init, T_INIT)

			x_curr = x_init.copy()

			while ode.successful() and ode.t < t_final and not has_recovered(x_curr, x_eq):
				x_curr = ode.integrate(ode.t + DT)

			if not ode.successful():
				print 'integration failed'

			elif ode.t >= t_final:
				print 'failed to converge'
				print ode.t
				print np.abs(x_curr - x_eq)
				print constants.RT * np.log(CONC_FOLD_CONVERGENCE)
				print dx_dt(0, x_curr)

			else:
				print 'recovered by {} (expected ~{})'.format(ode.t, -EXPECTED_RECOVERY_EPOCHS/largest_real_eigenvalue)

			x_final.append(x_curr)

		import ipdb; ipdb.set_trace()


