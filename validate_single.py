
'''

Validates the dynamical viability of a single set of estimated parameter
values.  Call pattern:

python validate_simply.py parameters_to_validate.npy

This file should have behavior consistent with validate_model.py, but no
guarantees.

'''
from __future__ import division

import sys
import os.path as pa
from itertools import izip

import numpy as np
np.random.seed(33)

import matplotlib.pyplot as plt
import scipy.integrate

import problems
import constants
import equations
import structure
from utils.linalg import approx_jac

# Criterion constants

EQU_CONC_THRESHOLD = 1.001

CONC_FOLD_PERTURBATION = 2.0
CONC_FOLD_CONVERGENCE = 1.01

PERTURBATION_SCALE = constants.RT * np.log(CONC_FOLD_PERTURBATION)
CONVERGENCE_SCALE = constants.RT * np.log(CONC_FOLD_CONVERGENCE)
N_PERTURBATIONS = 30

APPROX_JAC_RADIUS = 1e-5

PERTURBATION_RECOVERY_TIME_TOLERANCE = 3
EXPECTED_RECOVERY_EPOCHS = np.log((CONC_FOLD_PERTURBATION - 1)/(CONC_FOLD_CONVERGENCE - 1))
PERTURBATION_RECOVERY_EPOCHS = PERTURBATION_RECOVERY_TIME_TOLERANCE * EXPECTED_RECOVERY_EPOCHS

TARGET_PYRUVATE_PRODUCTION = 0.14e-3

FLUX_FIT_THRESHOLD = 1e-3

# ODE integration parameters

DT = 1e1
T_INIT = 0
INTEGRATOR = 'lsoda'
INTEGRATOR_OPTIONS = dict(
	atol = 1e-6, # Default absolute tolerance is way too low (1e-12)
	# first_step = 0.1,
	# max_step = 1.0
	# rtol = 1e-6
	)

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

# pars = np.load('optimized_pars.npy')
# pars = np.load('out/all_scaled/seed-21/pars.npy')

pars = np.load(sys.argv[1])

def test(pars):
	dx_dt = lambda t, x: dg_dt(x, pars)

	x_start = init_dg_dt(pars)

	t_final = PERTURBATION_RECOVERY_EPOCHS / constants.MU

	ode = scipy.integrate.ode(dx_dt)

	ode.set_initial_value(x_start, T_INIT)

	ode.set_integrator(
		INTEGRATOR,
		**INTEGRATOR_OPTIONS
		)

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

	pars_final = structure.glc_association_matrix.T.dot(x_eq)
	pars_final[is_static] = pars[is_static]

	v = equations.reaction_rates(pars_final, *equations.args)

	net_pyruvate_production = v[-2] - v[-1]

	flux_fit = (net_pyruvate_production / TARGET_PYRUVATE_PRODUCTION - 1)**2

	flux_is_fit = (flux_fit < FLUX_FIT_THRESHOLD)

	print 'net pyruvate production:', net_pyruvate_production
	print 'flux rel error:', flux_fit
	print 'is flux fit?', flux_is_fit

	normed_log_conc_deviation = np.linalg.norm(x_eq - x_start, 2) / constants.RT

	if not ode.successful() or not (normed_log_conc_deviation < np.log(EQU_CONC_THRESHOLD)):
		equ = False
		lre = None
		stable = False

		if not ode.successful():
			print 'ODE integration failure'

		else:
			print 'Initial concentrations too far from equilibrium ({:0.2e}, should be < 1)'.format(
				normed_log_conc_deviation / np.log(EQU_CONC_THRESHOLD)
				)

		return equ, lre, stable

	else:
		equ = True

	# x_eq = init_dg_dt(pars)

	jac = approx_jac(lambda x: dx_dt(T_INIT, x), x_eq, APPROX_JAC_RADIUS)

	(eigvals, eigvecs) = np.linalg.eig(jac)

	largest_real_eigenvalue = np.max(np.real(eigvals))


	# print largest_real_eigenvalue / constants.MU

	lre = largest_real_eigenvalue

	if largest_real_eigenvalue >= 0:
		stable = False
		return equ, lre, stable

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

		ode.set_integrator(
			INTEGRATOR,
			# **INTEGRATOR_OPTIONS # TODO
			)

		x_curr = x_init.copy()

		# t_hist = []
		# x_hist = []

		while ode.successful() and ode.t < t_final and not np.linalg.norm(x_curr - x_eq, 2) < CONVERGENCE_SCALE:
			x_next = ode.integrate(ode.t + DT)

			# if np.any(np.isnan(x_next)):
			# 	import matplotlib.pyplot as plt

			# 	plt.plot(t_hist, x_hist)

			# 	plt.savefig('temp.pdf')

			# 	import ipdb; ipdb.set_trace()

			x_curr = x_next

			# t_hist.append(ode.t)
			# x_hist.append(x_curr)

		# print np.linalg.norm(x_curr - x_eq, 2) / CONVERGENCE_SCALE

		# x_final.append(x_curr)

		if not ode.successful():
			print p, 'integration failed'
			stable = False
			break

		elif ode.t >= t_final:
			print p, 'failed to converge'
			print ode.t
			print np.abs(x_curr - x_eq)
			print constants.RT * np.log(CONC_FOLD_CONVERGENCE)
			print dx_dt(0, x_curr)
			stable = False
			break

		# else:
		# 	print 'recovered by {} (expected ~{})'.format(ode.t, -EXPECTED_RECOVERY_EPOCHS/largest_real_eigenvalue)

	else:
		stable = True

	return equ, lre, stable

equ, lre, stable = test(pars)

print equ, lre, stable

# import ipdb; ipdb.set_trace()
