
from __future__ import division

import os.path as pa
from itertools import izip

import numpy as np
import matplotlib.pyplot as plt

import fitting

def load_problem(problem):
	obj = np.load(pa.join('out', problem, 'obj.npy'))
	pars = np.load(pa.join('out', problem, 'pars.npy'))
	equ = np.load(pa.join('out', problem, 'equ.npy'))
	stable = np.load(pa.join('out', problem, 'stable.npy'))
	lre = np.load(pa.join('out', problem, 'lre.npy'))
	valid = np.load(pa.join('out', problem, 'valid.npy'))

	return obj, pars, equ, stable, lre, valid

PROBLEM = (
	# 'data_agnostic'
	# 'Keq_scaled'
	# 'Keq_e-1'
	# 'Keq_half'
	# 'history/standard'
	# 'custom'
	# 'no_seof_gap_kinetics_100x'
	# 'no_seof_gap_kinetics_100x_scaled'
	# 'seof_1e-1'
	# 'no_seof'
	# 'Keq_FBA_TPI_GPM_e-1'
	'all_scaled'
	)

obj, pars, equ, stable, lre, valid = load_problem(PROBLEM)

N = 300
N_retained = np.inf

selected = (np.arange(N) < N_retained)
np.random.shuffle(selected)

pars = pars[:, valid & selected]

def restrict_datatype(*datatypes):
	return (
		(lambda entry: (entry.datatype in datatypes), 1),
		(lambda entry: True, 0)
		)

names_and_rulesets = {
	'concentration':restrict_datatype('concentration'),
	# 'seof':restrict_datatype('standard_energy_of_formation'),
	'kinetics':restrict_datatype(
		'forward_catalytic_rate',
		'reverse_catalytic_rate',
		'reactant_saturation',
		'product_saturation'
		),
	'equilibrium':restrict_datatype('equilibrium')
	}

import constants

from initialization import build_initial_parameter_values
import bounds

from problems import DEFINITIONS

rules_and_weights = DEFINITIONS[PROBLEM]

(fitting_matrix, fitting_values) = fitting.build_fitting_tensors(*rules_and_weights)[:2]
relative_fitting_tensor_sets = fitting.build_relative_fitting_tensor_sets(*rules_and_weights)

bounds_matrix = bounds.BOUNDS_MATRIX
lowerbounds = bounds.LOWERBOUNDS
upperbounds = bounds.UPPERBOUNDS

import structure

(init_pars, init_fitness) = build_initial_parameter_values(
	bounds_matrix, (lowerbounds + upperbounds)/2,
	np.concatenate([-bounds_matrix, +bounds_matrix]), np.concatenate([-lowerbounds, upperbounds]),
	fitting_matrix, fitting_values,
	np.zeros((0, structure.n_parameters)), np.zeros((0,)),
	# sr_ul_mat, sr_ul_values,
	*[(fm, fv) for (fm, fv, fe) in relative_fitting_tensor_sets]
	)

tenfold = constants.RT * np.log(1e1)
billfold = constants.RT * np.log(1e9)

# PERCENTILES = (0.05, 0.5, 0.95)
PERCENTILES = (0.25, 0.5, 0.75)

percentiles = []

ids = []

all_residuals = []
init_residuals = []

for (name, ruleset) in names_and_rulesets.viewitems():
	(fm, fv, fe) = fitting.build_fitting_tensors(*ruleset)

	residuals = fm.dot(pars) - fv[:, None]

	all_residuals.append(residuals)

	percentiles.append(
		np.percentile(residuals, 100 * np.array(PERCENTILES), 1)
		)

	ids.extend(e.id for e in fe)

	init_residuals.append(
		fm.dot(init_pars) - fv
		)

for (fm, fv, fe) in fitting.build_relative_fitting_tensor_sets():

	name = fe[0].source # assuming all the same source

	raw_residuals = fm.dot(pars) - fv[:, None]

	medians = np.median(raw_residuals, 0)

	residuals = raw_residuals - medians[None, :]

	all_residuals.append(residuals)

	percentiles.append(
		np.percentile(residuals, 100 * np.array(PERCENTILES), 1)
		)

	ids.extend(e.id for e in fe)

	init_raw_residuals = fm.dot(init_pars) - fv

	init_medians = np.median(init_raw_residuals)

	init_residuals.append(
		init_raw_residuals - init_medians
		)

all_residuals = np.concatenate(all_residuals, 0)
# import ipdb; ipdb.set_trace()

init_residuals = np.concatenate(init_residuals)

(left, middle, right) = np.concatenate(percentiles, 1)

temp = np.zeros(
	left.size,
	[('type_index', 'i'), ('rc', 'i'), ('source', 'a40')]
	)

datatype_indexing = {s:i for i, s in enumerate([
	'concentration',
	# 'standard_energy_of_formation',
	'forward_catalytic_rate',
	'reverse_catalytic_rate',
	'reactant_saturation',
	'product_saturation',
	'relative_protein_count',
	'equilibrium',
	])}

import structure
from data import kb

nonstandard_reactions = sorted([
	entry.id for entry in kb.reaction
	if entry.id not in structure.reactions
	])

all_reactions = structure.reactions + nonstandard_reactions

for i, id_ in enumerate(ids):
	split = id_.split(':')

	datatype = split[0]
	rc = split[1:-1]
	source = split[-1]

	datatype_index = datatype_indexing[datatype]

	if len(rc) == 1:
		if datatype in ('concentration', 'standard_energy_of_formation'):
			rc_index = structure.compounds.index(rc[0])

		elif datatype == 'equilibrium':
			rc_index = all_reactions.index(rc[0])

		else:
			rc_index = structure.reactions.index(rc[0])

	else:
		rc_index = structure.reactions.index(rc[0])*len(structure.compounds) + structure.compounds.index(rc[1])

	temp[i] = (datatype_index, rc_index, source)

sorting = np.argsort(temp)

minor_seps = []
major_seps = []

for i, values in enumerate(temp[sorting]):
	if i != 0:
		if (values['type_index'] != last_values['type_index']):
			major_seps.append(i)

		elif (values['rc'] != last_values['rc']):
			minor_seps.append(i)

	last_values = values

left = left[sorting]
middle = middle[sorting]
right = right[sorting]
init_residuals = init_residuals[sorting]
all_residuals = all_residuals[sorting]
ids = np.array(ids)[sorting]

# import ipdb; ipdb.set_trace()

fig = plt.figure(figsize = (1, 6), dpi = 300)

axes = fig.add_axes((0, 0, 1, 1))

LW = 0.5

axes.axvline(0, color = (0.5,)*3, lw = LW, zorder = -1)
axes.axvline(-tenfold, color = (0.8,)*3, lw = LW, zorder = -1)
axes.axvline(+tenfold, color = (0.8,)*3, lw = LW, zorder = -1)

for sep in major_seps:
	axes.axhline(
		-(sep-0.5),
		color = (0.5,)*3,
		lw = LW,
		zorder = -2
		)

for sep in minor_seps:
	axes.axhline(
		-(sep-0.5),
		color = (0.8,)*3,
		# ls = ':',
		lw = LW,
		zorder = -2
		)

y = -np.arange(middle.size)

axes.errorbar(middle, y, xerr = (middle-left, right-middle), fmt = '.', c = (0.2,)*3, lw = LW, ms = 2, zorder = 0)

axes.plot(init_residuals, y, 'o', c = (0.2,)*3, ms = 4, markeredgewidth = LW, zorder = 1, markerfacecolor = 'none')

with open('residuals_key.txt', 'w') as f:
	f.write('\n'.join(ids))

inacc = np.abs(middle) > tenfold
broad = (right-left) > 2*tenfold

print 'inaccurate median'
print '-'*79
for i in np.where(inacc)[0]:
	print ids[i]

print

print 'broad dist'
print '-'*79
for i in np.where(broad)[0]:
	print ids[i]

print (~(inacc | broad)).mean()
print inacc.size

for i, n in enumerate(1*inacc + broad):
	if n:
		plt.text(
			billfold-tenfold*1.5, -(i+0.15),
			'*'*n,
			va = 'center',
			ha = 'left',
			color = 'r',
			size = 5
			)

axes.set_xlim(-billfold, +billfold)
axes.set_ylim(-(middle.size-0.5), +0.5)

axes.axis('off')

# fig.savefig('residuals.pdf')
fig.savefig('residuals.png', dpi = 300)

print 'average fit: {:0.2f}'.format(np.mean(np.sum(np.abs(all_residuals), 0)))
