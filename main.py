
from __future__ import division

import os
import os.path

import argparse

import numpy as np

import problems
import optimize

# TODO: move this logic to __main__ in optimize.py?

parser = argparse.ArgumentParser()
parser.add_argument(
	'--problem', type = str, default = 'all_scaled'
	)
parser.add_argument(
	'--seed', type = int, default = None
	)
parser.add_argument(
	'--n', type = int, default = 1
	)
parser.add_argument(
	'--force', type = bool, default = False
	)

args = parser.parse_args()

try:
	rules_and_weights = problems.DEFINITIONS[args.problem]

except KeyError:
	raise Exception('Unknown problem "{}".'.format(args.problem))

print 'Using problem definition "{}".'.format(args.problem)

for seed_offset in xrange(args.n):

	if args.seed is None:
		random_state = np.random.RandomState(None)
		print 'No seed provided; a random seed will be used.'
		print 'Output will not be saved.'
		outdir = None

	else:
		seed = args.seed + seed_offset

		random_state = np.random.RandomState(seed)
		print 'Using random seed {}.'.format(seed)
		outdir = os.path.join('out', args.problem, 'seed-{}'.format(seed))

		print 'Output will be saved to {}.'.format(os.path.abspath(outdir))
		try:
			os.makedirs(outdir)

		except OSError:
			assert os.path.exists(outdir)

		pars_path = os.path.join(outdir, 'pars.npy')
		obj_path = os.path.join(outdir, 'obj.npy')

	if outdir is None or args.force or not os.path.exists(pars_path) or not os.path.exists(obj_path):

		(pars, obj) = optimize.estimate_parameters(rules_and_weights, random_state)

		if outdir is not None:
			np.save(pars_path, pars)
			np.save(obj_path, np.array([
				obj.mass_eq,
				obj.energy_eq,
				obj.flux,
				obj.fit
				]))

	else:
		print 'Skipped - output already exists'
