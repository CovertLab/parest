
from __future__ import division

import numpy as np

import matplotlib.pyplot as plt

from constants import MU

def load_lre(path):
	valid = np.load(path + 'valid.npy')
	lre = np.load(path + 'lre.npy')

	return lre[valid]

def load_pars(path):
	valid = np.load(path + 'valid.npy')
	pars = np.load(path + 'pars.npy')

	return pars[:, valid]

plt.figure(figsize = (6, 6))

sources = [
	'out/all_scaled/',
	'out/all_scaled_upper_sat_limits_1e-1/',
	'out/all_scaled_upper_sat_limits_1e0/',
	'out/all_scaled_upper_sat_limits_1e1/',
	'out/all_scaled_upper_sat_limits_1e2/',
	]

COLORS = [
	np.array((225, 6, 133), np.float64)/255.,
	np.array((83, 49, 0), np.float64)/255.,
	np.array((143, 85, 0), np.float64)/255.,
	np.array((221, 145, 23), np.float64)/255.,
	np.array((251, 177, 37), np.float64)/255.,
	]

lre = [
	np.log10(-MU/load_lre(source))
	for source in sources
	]

bottom = -3
top = 0

x_range = top - bottom

nbins = 40

width = x_range/nbins
x = np.linspace(bottom, top, nbins, endpoint = False) + width/2
width_ratio = 1.00

bin_sets = []
median_bins = []
selected = []

for values in lre:
	bins = np.zeros(nbins, np.float64)

	median_bin = np.int64((np.median(values)-bottom)/(top-bottom) * nbins)

	median_bins.append(median_bin)

	select = None

	for index, value in enumerate(values):
		i = np.int64((value - bottom)/(top - bottom) * nbins)

		if (i < 0) or (i >= nbins):
			raise Exception('invalid index {} for value {}'.format(i, value))

		bins[i] += 1

		if select is None and i == median_bin:
			select = index

	selected.append(select)

	bins /= bins.sum()

	bin_sets.append(bins)

from matplotlib.ticker import FixedLocator

minor_offsets = np.array([np.log10(i) for i in xrange(2, 10)])

for i in xrange(len(sources)):
	plt.subplot(len(sources), 2, i*2+1)

	for index, bins in enumerate(bin_sets):
		if index != i:
			continue

		plt.bar(
			x,
			bins,
			width = width*width_ratio,
			zorder = 2 if (index == i) else 0,
			alpha = 1 if (index == i) else 0.2,
			color = COLORS[index]
			)

		if index == i:
			plt.plot(
				median_bins[i]*(top - bottom)/nbins + width/2 + bottom,
				0.05,
				'*',
				c = 'w',
				ms = 3,
				markeredgewidth =0
				)

			plt.title('n = {}'.format(lre[index].size))

	plt.yticks([])

	# plt.xticks(*zip(*(
	# 	(j, r'$10^{{}}$'.format(j) if (i == len(sources)-1) else '')
	# 	for j in xrange(bottom, top+1)
	# 	)))

	plt.gca().xaxis.set_minor_locator(FixedLocator(np.concatenate([
		minor_offsets + j for j in xrange(bottom, top+1)
		])))

	plt.xlim(bottom - x_range*0.05, top + x_range*0.05)

# plt.xlabel(r'$-\mathfrak{R}\left[\lambda\right]_\max$, in multiples of $\mu$')

import constants
import equations
import structure
import scipy.integrate
def plot_prd(pars, seed = None):
	SEED_OFFSET = 0

	CONC_FOLD_PERTURBATION = 2
	CONC_FOLD_CONVERGENCE = 1.01

	PERTURBATION_SCALE = constants.RT * np.log(CONC_FOLD_PERTURBATION)
	# CONVERGENCE_SCALE = constants.RT * np.log(CONC_FOLD_CONVERGENCE)

	PERTURBATION_RECOVERTY_TIME_TOLERANCE = 3
	EXPECTED_RECOVERY_EPOCHS = np.log((CONC_FOLD_PERTURBATION - 1)/(CONC_FOLD_CONVERGENCE - 1))
	PERTURBATION_RECOVERY_EPOCHS = PERTURBATION_RECOVERTY_TIME_TOLERANCE * EXPECTED_RECOVERY_EPOCHS

	DT = 1e0
	T_INIT = 0
	INTEGRATOR = 'lsoda'
	INTEGRATOR_OPTIONS = dict(
		atol = 1e-6 # Default absolute tolerance is way too low (1e-12)
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

	dx_dt = lambda t, x: dg_dt(x, pars)

	x_start = init_dg_dt(pars)

	t_final = PERTURBATION_RECOVERY_EPOCHS / constants.MU

	ode = scipy.integrate.ode(dx_dt)

	ode.set_initial_value(x_start, T_INIT)

	ode.set_integrator(INTEGRATOR, **INTEGRATOR_OPTIONS)

	while ode.successful() and ode.t < t_final:
		x_curr = ode.integrate(ode.t + DT)

	x_eq = x_curr

	EQU_CONC_THRESHOLD = 1.5

	assert ode.successful()
	assert (
		np.linalg.norm(x_eq - x_start, 2)
		< constants.RT * np.log(EQU_CONC_THRESHOLD)
		)

	t_final = 1/constants.MU

	perturbation = np.random.RandomState(
		seed + SEED_OFFSET
		).normal(size = x_eq.size)
	perturbation /= np.linalg.norm(perturbation, 2)
	perturbation *= PERTURBATION_SCALE

	x_init = x_eq + perturbation

	ode = scipy.integrate.ode(dx_dt)

	ode.set_initial_value(x_init, T_INIT)

	ode.set_integrator(INTEGRATOR)

	t_hist = [T_INIT]
	x_hist = [x_init.copy()]

	while ode.successful() and ode.t < t_final: # and not np.linalg.norm(x_hist[-1] - x_eq, 2) < CONVERGENCE_SCALE:
		new_t = ode.t + DT

		t_hist.append(new_t)
		x_hist.append(ode.integrate(new_t))

	t_hist = np.array(t_hist)
	x_hist = np.array(x_hist)

	# dc_pyr = []

	# for glc in x_hist:
	# 	x = structure.glc_association_matrix.T.dot(glc)
	# 	x[is_static] = pars[is_static]

	# 	v = equations.reaction_rates(x, *equations.args)

	# 	dc_pyr.append(
	# 		v[-2] - v[-1]
	# 		)

	x_diff = x_hist - x_eq

	c2 = x_diff / (constants.RT * np.log(2))

	# r = rdp(np.column_stack([t_hist, c2]), 1e-3)

	# print r.mean()

	for i, c in enumerate(structure.DYNAMIC_COMPOUNDS):
		# plt.plot(t_hist[r] * constants.MU, c2[r, i], label = c, lw = 2)
		plt.plot(t_hist, c2[:, i], label = c, lw = 2.0, color = 'w')
		plt.plot(t_hist, c2[:, i], label = c, lw = 1.5, color = COLORS[seed])

	# plt.ylim(0.5, 2)
	# plt.yticks([])

	# plt.xticks(np.arange(6))
	# plt.yticks([-1, 0, +1], [r'$\frac{1}{2}$x', '1x', '2x'])

	plt.xticks([])
	plt.yticks([])

	# plt.axhline(0, c = [0.5]*3, ls = '-', lw = 0.5, zorder = -1)
	# plt.axvline(0, c = [0.5]*3, ls = '-', lw = 0.5, zorder = -1)

	# plt.xlim(-0.3, +5.3)
	# plt.xlim(-0.05, +1.05)

	WIDTH = 1/constants.MU

	plt.xlim(-0.05 * WIDTH, 1.05 * WIDTH)
	# plt.ylim(-1.1, +1.1)

	YLIM = +1

	plt.ylim(-1, +1)

	# plt.legend(loc = 'best', fontsize = 5)

	# plt.title('Small perturbation recovery')
	# plt.xlabel(r'time, normalized by $\mu$')
	# plt.ylabel(r'deviation from equillibrium concentrations')

	# plt.yscale('symlog', linthreshy = 0.01)

	plt.xticks([0, WIDTH])

	plt.axhline(0, lw = 0.5, color = 'k', zorder = -10)
	plt.axvline(0, lw = 0.5, color = 'k', zorder = -10)

	# plt.title('{:0.1e} unrecovered'.format(np.linalg.norm(x_diff[-1, :], 2) / PERTURBATION_SCALE))

for i, source in enumerate(sources):
	pars = load_pars(source)[:, selected[i]]

	plt.subplot(len(sources), 2, i*2+2)
	plot_prd(pars, i)

plt.savefig('figure4.pdf')
