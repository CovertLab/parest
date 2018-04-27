
from __future__ import division

import os.path as pa
from itertools import izip

import sys

import numpy as np
np.random.seed(33)

import matplotlib.pyplot as plt
import scipy.integrate

import problems
import constants
import equations
import structure
from utils.linalg import approx_jac

FORCE = True

DRY_RUN = False

EQU_CONC_THRESHOLD = 1.001

CONC_FOLD_PERTURBATION = 2
CONC_FOLD_CONVERGENCE = 1.01

PERTURBATION_SCALE = constants.RT * np.log(CONC_FOLD_PERTURBATION)
CONVERGENCE_SCALE = constants.RT * np.log(CONC_FOLD_CONVERGENCE)
N_PERTURBATIONS = 30

APPROX_JAC_RADIUS = 1e-5

PERTURBATION_RECOVERTY_TIME_TOLERANCE = 3
EXPECTED_RECOVERY_EPOCHS = np.log((CONC_FOLD_PERTURBATION - 1)/(CONC_FOLD_CONVERGENCE - 1))
PERTURBATION_RECOVERY_EPOCHS = PERTURBATION_RECOVERTY_TIME_TOLERANCE * EXPECTED_RECOVERY_EPOCHS

TARGET_PYRUVATE_PRODUCTION = 0.14e-3
FLUX_FIT_THRESHOLD = 1e-3

DT = 1e1
T_INIT = 0
INTEGRATOR = 'lsoda'
INTEGRATOR_OPTIONS = dict(
	atol = 1e-6 # Default absolute tolerance is way too low (1e-12)
	)

def load_pars(target_dir):
	obj = np.load(pa.join(target_dir, 'obj.npy'))
	pars = np.load(pa.join(target_dir, 'pars.npy'))

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

target_dir = sys.argv[1]

stable_path = pa.join(target_dir, 'stable.npy')
equ_path = pa.join(target_dir, 'equ.npy')
lre_path = pa.join(target_dir, 'lre.npy')
valid_path = pa.join(target_dir, 'valid.npy')

paths = [stable_path, equ_path, lre_path, valid_path]

if not FORCE and all(pa.exists(path) for path in paths):
	print 'Skipping {}, output already exists'.format(target_dir)

else:
	print 'Validating {} parameters'.format(target_dir)

	(all_obj, all_pars) = load_pars(target_dir)

	equ = []
	lre = []
	stable = []

	for (i, pars) in enumerate(all_pars.T):
		# print '-'*79
		# print i, all_obj[:, i]

	# for pars in [all_pars[:, 214]]:

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

		normed_log_conc_deviation = np.linalg.norm(x_eq - x_start, 2) / constants.RT

		if not ode.successful() or not (normed_log_conc_deviation < np.log(EQU_CONC_THRESHOLD)):
			equ.append(False)
			lre.append(None)
			stable.append(False)
			print 'BAD'
			continue

		else:
			equ.append(flux_is_fit)

			if not flux_is_fit:
				v_init = equations.reaction_rates(pars, *equations.args)
				print 'flux error:', net_pyruvate_production, v_init[-2] - v_init[-1]

		# if not flux_is_fit: print 'bad flux: {}'.format(net_pyruvate_production)

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

		print i, np.mean(equ), np.mean(stable)

	stable = np.array(stable, np.bool)
	equ = np.array(equ, np.bool)
	lre = np.array(lre, np.float64)

	valid = (stable & equ)

	if not DRY_RUN:
		np.save(stable_path, stable)
		np.save(equ_path, equ)
		np.save(lre_path, lre)
		np.save(valid_path,valid)
