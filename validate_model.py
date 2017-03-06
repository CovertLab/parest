
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

PERTURBATION_SCALE = constants.RT * np.log(CONC_FOLD_PERTURBATION)
CONVERGENCE_SCALE = constants.RT * np.log(CONC_FOLD_CONVERGENCE)
N_PERTURBATIONS = 30

APPROX_JAC_RADIUS = 1e-5

PERTURBATION_RECOVERTY_TIME_TOLERANCE = 3
EXPECTED_RECOVERY_EPOCHS = np.log((CONC_FOLD_PERTURBATION - 1)/(CONC_FOLD_CONVERGENCE - 1))
PERTURBATION_RECOVERY_EPOCHS = PERTURBATION_RECOVERTY_TIME_TOLERANCE * EXPECTED_RECOVERY_EPOCHS

DT = 1e1
T_INIT = 0
INTEGRATOR = 'lsoda'

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

# data_agnostic, 67 & 214 - high initial concentration of F6P leads to NaN errors

for problem in ['data_agnostic']: #problems.DEFINITIONS.viewkeys():
	(all_obj, all_pars) = load_pars(problem)

	equ = []
	lre = []
	stable = []

	for (i, pars) in enumerate(all_pars.T):
		# print '-'*79
		# print i, all_obj[:, i]

		print i

	# for pars in [all_pars[:, 214]]:

		dx_dt = lambda t, x: dg_dt(x, pars)

		x_start = init_dg_dt(pars)

		t_final = PERTURBATION_RECOVERY_EPOCHS / constants.MU

		ode = scipy.integrate.ode(dx_dt)

		ode.set_initial_value(x_start, T_INIT)

		ode.set_integrator(INTEGRATOR)

		# x_hist = [x_start.copy()]

		while ode.successful() and ode.t < t_final:
			x_curr = ode.integrate(ode.t + DT)
			# TODO: terminate when x_curr stops changing

			# x_hist.append(ode.integrate(ode.t + DT))

		# x_curr = x_hist[-1]

		# all_x = np.array(x_hist)

		# f = plt.figure()
		# plt.plot(all_x - x_curr[None, :])
		# plt.savefig('ode.pdf')

		x_eq = x_curr

		EQU_CONC_THRESHOLD = 1.5

		if not ode.successful() or not (np.linalg.norm(x_eq - x_start, 2) < constants.RT * np.log(EQU_CONC_THRESHOLD)):
			equ.append(False)
			lre.append(None)
			stable.append(False)
			continue

		else:
			equ.append(True)

		# import ipdb; ipdb.set_trace()

		# x_eq = init_dg_dt(pars)

		jac = approx_jac(lambda x: dx_dt(T_INIT, x), x_eq, APPROX_JAC_RADIUS)

		(eigvals, eigvecs) = np.linalg.eig(jac)

		largest_real_eigenvalue = np.max(np.real(eigvals))

		# import ipdb; ipdb.set_trace()

		# print largest_real_eigenvalue / constants.MU

		lre.append(largest_real_eigenvalue)

		if largest_real_eigenvalue >= 0:
			stable.append(False)
			continue

		t_final = -PERTURBATION_RECOVERY_EPOCHS / largest_real_eigenvalue

		# x_final = []

		for p in xrange(N_PERTURBATIONS):
			# x_init = x_eq + (np.random.uniform(size = x_eq.size) - 0.5) * PERTURBATION_SCALE
			perturbation = np.random.normal(size = x_eq.size)
			perturbation /= np.linalg.norm(perturbation, 2)
			perturbation *= PERTURBATION_SCALE

			x_init = x_eq + perturbation

			ode = scipy.integrate.ode(dx_dt)

			ode.set_initial_value(x_init, T_INIT)

			ode.set_integrator(INTEGRATOR)

			x_curr = x_init.copy()

			while ode.successful() and ode.t < t_final and not np.linalg.norm(x_curr - x_eq, 2) < CONVERGENCE_SCALE:
				x_curr = ode.integrate(ode.t + DT)

			# print np.linalg.norm(x_curr - x_eq, 2) / CONVERGENCE_SCALE

			# x_final.append(x_curr)

			if not ode.successful():
				# print 'integration failed'
				stable.append(False)
				break

			elif ode.t >= t_final:
				# print 'failed to converge'
				# print ode.t
				# print np.abs(x_curr - x_eq)
				# print constants.RT * np.log(CONC_FOLD_CONVERGENCE)
				# print dx_dt(0, x_curr)
				stable.append(False)
				break

			# else:
			# 	print 'recovered by {} (expected ~{})'.format(ode.t, -EXPECTED_RECOVERY_EPOCHS/largest_real_eigenvalue)

		else:
			stable.append(True)

		# import ipdb; ipdb.set_trace()

import ipdb; ipdb.set_trace()
